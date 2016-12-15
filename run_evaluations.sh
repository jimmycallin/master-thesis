source activate py2

trap "echo Exited!; exit;" SIGINT SIGTERM

for f in outputs/*; do
	test_data=$(echo $f | grep -o "conll.*$" -)
	name=$(echo $f | sed -e "s#^outputs/##")
    if [ -z "$test_data" ]; then
        continue
    fi
    if [ -f "results/$name.prototext" ]; then
    	continue
    fi
    echo "python utils/conll16st-eval/tira_sup_eval.py \"resources/conll16st-en-zh-dev-train-test_LDC2016E50/$test_data\" \"$f\" \"results/$name.prototext\""
	python utils/conll16st-eval/tira_sup_eval.py "resources/conll16st-en-zh-dev-train-test_LDC2016E50/$test_data" "$f" "results/$name.prototext"
	if ! [ -s "results/$name.prototext" ]; then
		echo "File results/$name.prototext is empty, removing"
		rm "results/$name.prototext"
	fi
done
