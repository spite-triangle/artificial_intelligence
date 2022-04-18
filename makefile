build:
	find ./ReinforcementLearning/chapter/ -name "*_withNum.md" -exec rm {} \;
	find ./ReinforcementLearning/chapter/ -name "*.md" -exec python3 ./AutoNum.py {} \;
	find ./ComputerVision/chapter/ -name "*_withNum.md" -exec rm {} \;
	find ./ComputerVision/chapter/ -name "*.md" -exec python3 ./AutoNum.py {} \;
	# sed -i 's/src="..\/..\//src="\/artificial_intelligence\//g' ./ReinforcementLearning/chapter/*_withNum.md
	# sed -i 's/src="..\/..\//src="\/artificial_intelligence\//g' ./ComputerVision/chapter/*_withNum.md
