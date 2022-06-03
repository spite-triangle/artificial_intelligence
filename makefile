build:
	find ./ReinforcementLearning/chapter/ -name "*_withNum.md" -exec rm {} \;
	find ./ReinforcementLearning/chapter/ -name "*.md" -exec python3 ./AutoNum.py {} \;
	find ./ComputerVision/chapter/ -name "*_withNum.md" -exec rm {} \;
	find ./ComputerVision/chapter/ -name "*.md" -exec python3 ./AutoNum.py {} \;
	find ./ComputerVision/practice/ -name "*_withNum.md" -exec rm {} \;
	find ./ComputerVision/practice/ -name "*.md" -exec python3 ./AutoNum.py {} \;
	find ./yolo/chapter/ -name "*_withNum.md" -exec rm {} \;
	find ./yolo/chapter/ -name "*.md" -exec python3 ./AutoNum.py {} \;
	find ./pytorch/chapter/ -name "*_withNum.md" -exec rm {} \;
	find ./pytorch/chapter/ -name "*.md" -exec python3 ./AutoNum.py {} \;
	sed -i 's/src="..\/..\//src="\/artificial_intelligence\//g' ./ReinforcementLearning/chapter/*_withNum.md
	sed -i 's/src="..\/..\//src="\/artificial_intelligence\//g' ./ComputerVision/chapter/*_withNum.md
	sed -i 's/src="..\/..\//src="\/artificial_intelligence\//g' ./ComputerVision/practice/*_withNum.md
	sed -i 's/src="..\/..\//src="\/artificial_intelligence\//g' ./yolo/chapter/*_withNum.md
	sed -i 's/src="..\/..\//src="\/artificial_intelligence\//g' ./pytorch/chapter/*_withNum.md
	cd  ./ComputerVision/chapter/ ; rm README_withNum.md _sidebar_withNum.md
	cd  ./ComputerVision/practice/ ; rm README_withNum.md _sidebar_withNum.md
