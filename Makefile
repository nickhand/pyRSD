.PHONY: all clean

all:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f