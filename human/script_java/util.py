import jsonlines

filename = "java_default_beam.json"
with open(filename) as f, open("java_default_beam.txt","w") as g:
    for line in jsonlines.Reader(f):
        g.write(line["predictions"][0]+'\n')