import sys
import onnx
import json

def dump_normal(elem, indent, file) :
    for s in str(elem).splitlines() :
        print(indent + s, file=file)

def dump_initializer(elem, indent, file) : 
    # calculate size.
    size = 1
    for d in elem.dims :
        size *= d

    # in the case of enough small size, output all data.
    if (size <= 32) :
        dump_normal(elem, indent, file)
        return

    # output metadata only, in all other cases.
    for d in elem.dims :
        print(indent + "  dims: " + json.dumps(d), file=file)
    print(indent + "  data_type: " + json.dumps(elem.data_type), file=file)
    print(indent + "  name: " + json.dumps(elem.name), file=file)

def onnx2prototxt(onnx_path) :

    # show information
    out_path = onnx_path + ".prototxt"
    print("+ creating " + out_path)
    print("    from " + onnx_path + " ...")

    # load model
    model = onnx.load(onnx_path)

    # print prototxt
    with open(out_path, "w") as f :
        print("ir_version: " + json.dumps(model.ir_version), file=f)
        print("producer_name: " + json.dumps(model.producer_name), file=f)
        print("producer_version: " + json.dumps(model.producer_version), file=f)
        # print("domain: " + json.dumps(model.domain), file=f)
        print("model_version: " + json.dumps(model.model_version), file=f)
        # print("doc_string: " + json.dumps(model.doc_string), file=f)
        print("graph {", file=f)
        print("  name: " + json.dumps(model.graph.name), file=f)

        for e in model.graph.node :
            print("  node {", file=f)
            dump_normal(e, "    ",  f)
            print("  }", file=f)

        for e in model.graph.initializer :
            print("  initializer {", file=f)
            dump_initializer(e, "    ",  f)
            print("  }", file=f)

        for e in model.graph.input :
            print("  input {", file=f)
            dump_normal(e, "    ",  f)
            print("  }", file=f)

        for e in model.graph.output :
            print("  output {", file=f)
            dump_normal(e, "    ",  f)
            print("  }", file=f)

        print("}", file=f)

        for e in model.opset_import :
            print("opset_import {", file=f)
            print("  version: " + json.dumps(e.version), file=f)
            print("}", file=f)


def show_usage(script) :
    print("usage: python " + script + " input.onnx [more.onnx ..]")


def main() : 
    if len(sys.argv) == 1 :
        show_usage(sys.argv[0])
        return

    for i in range(1,len(sys.argv)) :
        onnx2prototxt(sys.argv[i])


if __name__ == "__main__":
    main()

