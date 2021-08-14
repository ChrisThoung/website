.PHONY : run
run : example_model.py data.txt
	python example_model.py

data.txt : process_raw_data.py SUT-Agg-98-17.xlsx
	python process_raw_data.py

SUT-Agg-98-17.xlsx :
	curl -s -o SUT-Agg-98-17.xlsx https://www.gov.scot/binaries/content/documents/govscot/publications/statistics/2019/08/input-output-latest/documents/aggregate-tables-all-years/aggregate-tables-all-years/govscot%3Adocument/SUT-Agg-98-17.xlsx

.PHONY : test
test :
	pytest -x -v --cache-clear --cov=iomodel --cov-branch && coverage report -m && coverage html

.PHONY : clean
clean :
	rm -f SUT-Agg-98-17.xlsx data.txt
