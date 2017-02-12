if [[ "$OSTYPE" == "darwin"* ]]; then
    ./configure --prefix=$PREFIX --enable-shared --disable-fortran --disable-static \
                --enable-threads --enable-sse2
else
    ./configure --prefix=$PREFIX --enable-shared --disable-fortran \
                --enable-threads --enable-sse2
fi

make
make install