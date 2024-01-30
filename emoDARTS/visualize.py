""" Network architecture visualizer using graphviz """
import sys

import genotypes as gt


def plot(genotype, file_path, caption=None):
    try:
        import dot2tex
        from graphviz import Digraph
        """ make DAG plot and save to file_path as .png """
        edge_attr = {
            'fontsize': '20',
            'fontname': 'times'
        }
        node_attr = {
            'style': 'filled',
            'shape': 'rect',
            'align': 'center',
            'fontsize': '20',
            'height': '0.5',
            'width': '0.5',
            'penwidth': '2',
            'fontname': 'times'
        }
        g = Digraph(
            format='pdf',
            edge_attr=edge_attr,
            node_attr=node_attr,
            engine='dot')
        g.body.extend(['rankdir=LR'])

        # input nodes
        g.node("c_{t-2}", label="c_{t-2}", fillcolor='darkseagreen2')
        g.node("c_{t-1}", label="c_{t-1}", fillcolor='darkseagreen2')

        # intermediate nodes
        n_nodes = len(genotype)
        for i in range(n_nodes):
            g.node(str(i), fillcolor='lightblue')

        for i, edges in enumerate(genotype):
            for op, j in edges:
                if j == 0:
                    u = "c_{t-2}"
                elif j == 1:
                    u = "c_{t-1}"
                else:
                    u = str(j - 2)

                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")

        # output node
        g.node("c_{t}", label="c_{t}", fillcolor='palegoldenrod')
        for i in range(n_nodes):
            g.edge(str(i), "c_{t}", fillcolor="gray")

        # add image caption
        if caption:
            g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

        g.render(file_path, view=False)
    except ModuleNotFoundError as e:
        print(f"Could not draw graph. Module Not Found {str(e)}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("usage:\n python {} GENOTYPE".format(sys.argv[0]))

    genotype_str = sys.argv[1]
    try:
        genotype = gt.from_str(genotype_str)
    except AttributeError:
        raise ValueError("Cannot parse {}".format(genotype_str))

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
    plot(genotype.rnn, "rnn")
