
# USAGE: $ sh addheader.sh "col1 col2 col3" input.txt output.txt 0

# 1 HEADER
# 2 INPUT_FILE
# 3 OUTPUT_FILE

echo "Adding Header."
echo "$1" | cat - $2 > $3
