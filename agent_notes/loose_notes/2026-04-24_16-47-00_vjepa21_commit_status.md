# V-JEPA 2.1 commit status

The V-JEPA 2.1 encoder fork and its smoke notes landed in commit:

```text
513959de296720f1096149e831c878014bd75e1a Add precomputed feature implicit trainer
```

That commit contains the V-JEPA work from this session:

- `HuggingFaceVJEPAVideoEncoder`
- `TorchHubVJEPAVideoEncoder`
- explicit public Meta checkpoint loading for V-JEPA 2.1 TorchHub models
- tiny `vjepa2_1_vit_base_384` config
- real pretrained smoke note showing the full implicit-camera forward passed

It also includes adjacent precomputed/LTX feature-cache work. I did not rewrite
or split that commit after it appeared on the branch, because doing so would
rewrite a concurrent commit in a dirty shared workspace.
