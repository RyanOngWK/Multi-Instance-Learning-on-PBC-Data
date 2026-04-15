import argparse
from src.utils import init_caches, seed_everything, ensure_dir, warm_up_cache
from src.models import build_encoder, AttentionMIL
from src.dataset import FungaiBagDataset, load_splits_from_csv, build_transforms
from src.engine import train_one_epoch, evaluate
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=False)
    parser.add_argument("--val-dir", type=str, required=False)
    parser.add_argument("--test-dir", type=str, required=False)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tiles-per-image", type=int, default=24)
    parser.add_argument("--sampling-stride", type=int, default=112)
    parser.add_argument("--min-fg-ratio", type=float, default=0.1)
    parser.add_argument("--keep-background-ratio", type=float, default=0.0)

    parser.add_argument("--attn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--encoder-checkpoint", type=str, required=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--original-foundation-checkpoint", type=str)
    parser.add_argument(
    "--encoder-type",
    type=str,
    choices=["ruipath", "nextvit", "fastvit", "mambavision","dinov3vit","dinov3convnext"],
    help="Which encoder backbone to use.",required=True
    )
    parser.add_argument(
    "--encoder-model-name",
    type=str,
    default=None,
    help="Exact timm/HF model string, e.g. 'nextvit_base', 'fastvit_t8'. "
         "If not set, a sensible default is used per encoder-type.",
    )
    parser.add_argument("--bags-csv",   default=None, help="Path to bags.csv with pre-assigned splits")
    parser.add_argument("--image-root", default="",   help="Local root folder to resolve image filenames from CSV")
    # ── NEW: optional on-disk mask cache ──────────────────────────────────
    parser.add_argument(
        "--mask-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory to persist foreground masks on disk. "
            "Speeds up restarts: masks are reloaded instead of recomputed. "
            "Defaults to <output-dir>/mask_cache when not specified."
        ),
    )
    # ── NEW: warm-up thread count ─────────────────────────────────────────
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=4,
        help="Number of threads used for the pre-epoch mask warm-up pass.",
    )
    parser.add_argument("--test-only", action="store_true", help="Skip training, load best_model.pt and evaluate on test set only.")    
    parser.add_argument("--train-json", type=str, default=None)
    parser.add_argument("--val-json",   type=str, default=None)
    parser.add_argument("--test-json",  type=str, default=None)

    return parser.parse_args()
def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    seed_everything(args.seed)

    # Default mask cache lives inside the output directory.
    mask_cache_dir = args.mask_cache_dir or os.path.join(args.output_dir, "mask_cache")

    # Initialise module-level cache singletons before any Dataset is created.
    init_caches(mask_cache_dir=mask_cache_dir)

    print("[INFO] Scanning train/test folders...")
    # class_map = build_class_map(args.train_dir, args.val_dir, args.test_dir)
    # train_df = scan_split_directory(args.train_dir, class_map)
    # val_df = scan_split_directory(args.val_dir, class_map)
    # test_df = scan_split_directory(args.test_dir, class_map)
    if args.bags_csv:
        train_df, val_df, test_df, class_map = load_splits_from_csv(
            args.bags_csv, image_root=args.image_root or ""
        )
    elif args.train_json:
        import json as _json
        train_df, val_df, test_df, class_map = load_splits_from_json(
            args.train_json, args.val_json, args.test_json, args.image_root
        )
    else:
        # existing folder-based logic...
        class_map = build_class_map(args.train_dir, args.val_dir, args.test_dir)
        train_df = scan_split_directory(args.train_dir, class_map)
        val_df   = scan_split_directory(args.val_dir,   class_map)
        test_df  = scan_split_directory(args.test_dir,  class_map)
    num_classes = len(class_map)
    print(f"[INFO] Num classes:   {num_classes}")
    print(f"[INFO] Train images:  {len(train_df)}")
    print(f"[INFO] Val images:    {len(val_df)}")
    print(f"[INFO] Test images:   {len(test_df)}")
    print("[INFO] Train label counts:")
    print(train_df["label_id"].value_counts().sort_index().to_string())
    print("[INFO] Val label counts:")
    print(val_df["label_id"].value_counts().sort_index().to_string())
    print("[INFO] Test label counts:")
    print(test_df["label_id"].value_counts().sort_index().to_string())

    save_dataframe(train_df, os.path.join(args.output_dir, "train_metadata.csv"))
    save_dataframe(val_df, os.path.join(args.output_dir, "val_metadata.csv"))
    save_dataframe(test_df, os.path.join(args.output_dir, "test_metadata.csv"))

    import json
    with open(os.path.join(args.output_dir, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    train_tf, test_tf = build_transforms()

    # ── datasets ──────────────────────────────────────────────────────────
    ds_kwargs = dict(
        tile_size=args.tile_size,
        tiles_per_image=args.tiles_per_image,
        min_fg_ratio=args.min_fg_ratio,
        stride=args.sampling_stride,
        keep_background_ratio=args.keep_background_ratio,
    )
    train_ds = FungaiBagDataset(train_df, transform=train_tf, **ds_kwargs)
    val_ds   = FungaiBagDataset(val_df,   transform=test_tf,  **ds_kwargs)
    test_ds  = FungaiBagDataset(test_df,  transform=test_tf,  **ds_kwargs)

    # ── warm-up: pre-compute all masks + coord lists ───────────────────────
    print("[INFO] Warming up mask / tile-coord caches...")
    warm_up_cache([train_ds, val_ds, test_ds], max_workers=args.cache_workers)
    print("[INFO] Cache warm-up complete.")

    # ── data loaders ──────────────────────────────────────────────────────
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        # Keep worker processes alive between epochs — avoids re-fork overhead.
        persistent_workers=(args.num_workers > 0),
        # Two pre-fetched batches per worker keeps GPU fed.
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # ── model ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # patch_encoder = RuiPathViTL16Encoder(checkpoint_path=args.encoder_checkpoint)
    print(f"[INFO] Encoder type: {args.encoder_type}")
    patch_encoder = build_encoder(
        encoder_type=args.encoder_type,
        checkpoint_path=args.encoder_checkpoint,
        model_name=args.encoder_model_name,
    )
    print(f"[INFO] Embedding dim: {patch_encoder.embedding_dim}")
    model = AttentionMIL(
        patch_encoder=patch_encoder,
        embedding_dim=patch_encoder.embedding_dim,
        num_classes=num_classes,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
    ).to(device)
    # checkpoint = torch.load(args.encoder_checkpoint, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"]) 
    # print("Full MIL model weights loaded successfully!")
    if args.freeze_encoder:
        for p in model.patch_encoder.parameters():
            p.requires_grad = False

    class_counts = train_df["label_id"].value_counts().sort_index()
    counts = np.array([class_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    counts[counts == 0] = 1.0
    class_weights = counts.sum() / (num_classes * counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.BCEWithLogitsLoss()# for malaria


    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── training loop ─────────────────────────────────────────────────────
    best_f1 = -math.inf
    history = []
    if not args.test_only:

        for epoch in trange(1, args.epochs + 1, desc="Epochs", leave=True):
            print("Starting Training")
            start = time.time()

            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics , _, _= evaluate(
                model,
                val_loader,
                criterion,
                device,
                heatmap_dir=None,
                tile_size=args.tile_size,
            )
            elapsed = time.time() - start

            row = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_acc": train_metrics.acc,
                "train_f1_macro": train_metrics.f1_macro,
                "val_loss": val_metrics.loss,
                "val_acc": val_metrics.acc,
                "val_f1_macro": val_metrics.f1_macro,
                "seconds": elapsed,
            }
            history.append(row)

            print(
                f"[Epoch {epoch:02d}] "
                f"train_loss={train_metrics.loss:.4f} "
                f"train_acc={train_metrics.acc:.4f} "
                f"train_f1_macro={train_metrics.f1_macro:.4f} | "
                f"val_loss={val_metrics.loss:.4f} "
                f"val_acc={val_metrics.acc:.4f} "
                f"val_f1_macro={val_metrics.f1_macro:.4f} | "
                f"time={elapsed:.1f}s"
            )

            current_f1 = val_metrics.f1_macro
            if np.isnan(current_f1):
                current_f1 = -math.inf

            if current_f1 > best_f1:
                best_f1 = current_f1
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1_macro": best_f1,
                    "class_map": class_map,
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))
                print(f"[INFO] Saved new best model with val_f1_macro={best_f1:.4f}")

        history_df = pd.DataFrame(history)
        save_dataframe(history_df, os.path.join(args.output_dir, "history.csv"))
        plot_training_curves(history_df, args.output_dir)   # ← add this line
    del optimizer
    torch.cuda.empty_cache()
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[INFO] Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}.")
    model_name = Path(args.encoder_checkpoint).stem  # e.g. "ruipath_visionfoundation_v1.0"

    final_test_metrics, test_y_true, test_y_pred = evaluate(
        model, test_loader, criterion, device,
        heatmap_dir=(
        os.path.join(args.output_dir, f"test_heatmaps_{model_name}") if args.save_heatmaps else None
        ),
        tile_size=args.tile_size,
    )

    # Summary metrics CSV
    final_test_row = pd.DataFrame([{
        "split": "test",
        "loss": final_test_metrics.loss,
        "acc": final_test_metrics.acc,
        "precision_macro": final_test_metrics.precision_macro,
        "precision_weighted": final_test_metrics.precision_weighted,
        "recall_macro": final_test_metrics.recall_macro,
        "recall_weighted": final_test_metrics.recall_weighted,
        "f1_macro": final_test_metrics.f1_macro,
        "f1_weighted": final_test_metrics.f1_weighted,
    }])
    save_dataframe(final_test_row, os.path.join(args.output_dir, f"final_test_metrics_{args.encoder_type}.csv"))
    
    # Per-class report CSV + printed table
    print("\n[INFO] Per-class classification report:")
    save_classification_report(
        y_true=test_y_true,
        y_pred=test_y_pred,
        class_map=class_map,
        out_path=os.path.join(args.output_dir, f"final_test_classification_report_{args.encoder_type}.csv"),
    )
    
    print("[INFO] Training complete.")
    print(f"[INFO] Best val F1-macro:       {best_f1:.4f}")
    print(f"[INFO] Test Loss:         {final_test_metrics.loss:.4f}")
    print(f"[INFO] Test Accuracy:          {final_test_metrics.acc:.4f}")
    print(f"[INFO] Test Precision (Weighted):{final_test_metrics.precision_weighted:.4f}")
    print(f"[INFO] Test Precision (Macro):   {final_test_metrics.precision_macro:.4f}")
    print(f"[INFO] Test Recall (Weighted): {final_test_metrics.recall_weighted:.4f}")
    print(f"[INFO] Test Recall (Macro):    {final_test_metrics.recall_macro:.4f}")    
    print(f"[INFO] Test F1 (Weighted):  {final_test_metrics.f1_weighted:.4f}")
    print(f"[INFO] Test F1 (Macro):     {final_test_metrics.f1_macro:.4f}")

if __name__ == "__main__":
        
    sys.argv=[        "train_mil.py",
        "--bags-csv", "xx/xx",
        "--image-root", "xx/xx",   # folder where images actually live
        "--encoder-checkpoint","xx",
        "--output-dir", "xx",
        "--epochs", "30",        
        "--epochs", "30",
        "--batch-size", "8",
        "--num-workers", "0",
        "--tile-size", "224",
        "--attn-dim", "256",
        "--dropout", "0.25",
        "--save-heatmaps",
        "--freeze-encoder",
        "--encoder-type", "dinov3convnext",
        # "--test-only",
        "--weight-decay","1e-3",
        "--lr","1e-3"
        ]