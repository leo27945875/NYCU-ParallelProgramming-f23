all: default grade

default: page_rank.cpp main.cpp
	g++ -I../ -std=c++17 -fopenmp -O3 -o pr main.cpp page_rank.cpp ../common/graph.cpp ref_pr.a
grade: page_rank.cpp grade.cpp
	g++ -I../ -std=c++17 -fopenmp -O3 -o pr_grader grade.cpp page_rank.cpp ../common/graph.cpp ref_pr.a
clean:
	rm -rf pr pr_grader *~ *.*~
reset:
	make clean
	make
test:
	./pr /HW3/graphs/grid100x100.graph 
prof:
	gprof -l pr gmon.out -b > profiling_result