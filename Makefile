# Renders reproducibles. Los productos finales van a assets/; las cachés de Manim a build/ (desechable).

SCENES := SuperpositionScene SAEAnatomyScene AuxKAblationScene NeuronVsLatentScene SteeringScene ContextLengthScene
MANIM  := uv run manim --media_dir build/manim --progress_bar none

.PHONY: videos covers smoke clean

videos:  ## 6 escenas del artículo a 1080p60 → assets/videos/
	$(MANIM) -qh src/viz/manim_scenes.py $(SCENES)
	mkdir -p assets/videos
	cp build/manim/videos/manim_scenes/1080p60/*.mp4 assets/videos/

covers:  ## 5 portadas (cairo) + portada prisma en Manim → assets/cover/
	uv run --with umap-learn python src/viz/cover.py
	$(MANIM) -s src/viz/manim_cover.py CoverPrismScene
	cp build/manim/images/manim_cover/CoverPrismScene_ManimCE_*.png assets/cover/cover_B_prisma_manim.png

smoke:  ## entrenamiento corto en CPU con reanudación
	uv run python src/train_sae.py --smoke --checkpoint-dir /tmp/sae_smoke --resume auto --wandb-mode disabled

clean:
	rm -rf build
