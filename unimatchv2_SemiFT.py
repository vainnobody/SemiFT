from unimatchv2_peft import get_parser, main

import yaml


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg.setdefault("peft", {})
    cfg["peft"]["method"] = "semift"
    cfg["peft"].setdefault("target_modules", ["mlp"])
    cfg["peft"].setdefault("freeze_backbone", True)
    main(args, cfg)
