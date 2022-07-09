echo -e "Install Python modules"
pip install -r requirements.txt
echo -e "Done!"
cd WWLS
echo -e "Compile C++ code file"
python setup.py build_ext -i
echo -e "Done!"
cd ..
