"""This module contains an object to manipulate SCOPe classification.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "23s/02/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


# [I]
import igviz
# [N]
import networkx as net
import numpy as np

# [M]
from matplotlib import colormaps
# [P]
import plotly.graph_objects as go
# [S]
from statistics import median
# [T]
from tqdm import tqdm

# [M]
from matplotlib import colormaps

from sys import path

# Checking parent directory.
path.append("src/embeddings/")

from context_analyzer import PairewiseContextAnalyzer, center_context

class Scope:
    """An object to manipulate SCOPe classification.
    """

    def __init__(
        self,
        path: str,
        order_matrix: np.ndarray,
        unorder_matrix: np.ndarray,
        index_dict: "dict[str : int]"
    ) -> None:
        """Instantiate a `Scope` object.

        Parameters
        ----------
        path : `str`
            A path to a SCOPe classification data file.
        """
        # To go from domain to SCOPe classification code.
        self.classification: "dict[str: str]" = {}

        # Create the network.
        self.tree: net.Graph = net.Graph()
        self.tree.add_node("0")

        # To use viridis colour.
        cmap: object = colormaps["viridis"]
        self.viridis: object = cmap(np.linspace(0, 1, 5))
        self.color = [f"rgba{tuple(self.viridis[0])}"]

        # To obtain the number of leaves in a given nodes.
        self.size: "list[int]" = [-1]
        self.index: "dict[str : int]" = {"0": 0}
        self.absolute_size: "dict[str : int]" = {"0": 0}
        self.order_abs_size: "list[int]" = []

        # Labels.
        self.label: "list[str]" = []

        self.order_matrix: np.ndarray = order_matrix
        self.unorder_matrix: np.ndarray = unorder_matrix
        self.index_dict: "dict[str : int]" = index_dict

        self.matrix_index: "list[list[int]]" = [[]]

        # Parse a given SCOPe classification data file.
        with open(path, "r", encoding="utf-8") as file:
            for line in tqdm(list(file), "    PARSING SCOPe FILE"):
                # Skip the comment lines.
                if line[0] == "#":
                    continue

                split_line: "list[str]" = line.split()

                if len(split_line[2]) <= 5:
                    if split_line[2] not in self.absolute_size:
                        self.absolute_size[split_line[2]] = 0

                # Skip the line containing other things than domains.
                if split_line[3] == "-":
                    continue

                self.absolute_size["0"] += 1
                split_class: "list[str]" = split_line[2].split(".")

                for i in range(4):
                    normal_class: str = ".".join(split_class[:i + 1])

                    if normal_class not in self.absolute_size:
                        self.absolute_size[normal_class] = 1
                    else:
                        self.absolute_size[normal_class] += 1

                # Adding a domain to the classification dictionary.
                self.classification[split_line[3]] = split_line[2]

        self.order_abs_size += [self.absolute_size["0"]]

    def add_domain(self, domain: str) -> None:
        """Add a domain to the network graph.

        Parameters
        ----------
        domain : `str`
            The domain to add.
        """
        # To check if the domain is in the classification dictionary.
        if domain not in self.classification:
            print(f"[Warn##] Domain [{domain}] is not in the SCOPe "
                  "classification dictionary, skipping...")
            return None

        # Transform `a.1.1.1` in a list like `["a", "1", "1", "1"]`
        class_code: "str" = self.classification[domain].split(".")
        last: str = "0"
        self.size[0] += 1
        self.matrix_index[0] += [self.index_dict[domain]]
        self.label += [""]

        # To all SCOPe level of a given class like : a; a.1; a.1.1 and a.1.1.1
        for i in range(len(class_code)):
            # Check a level.
            node: str = ".".join(class_code[0:i + 1])

            # Is this a new node to add ?
            if node not in self.index:
                self.order_abs_size += [self.absolute_size[node]]

                if len(node) <= 1:
                    self.label += [node]
                else:
                    self.label += [""]

                self.size += [1]
                self.matrix_index += [[self.index_dict[domain]]]
                self.index[node] = len(self.size) - 1
            else:
                # Increase node's size.
                self.size[self.index[node]] += 1
                self.matrix_index[self.index[node]] += [self.index_dict[domain]]

            # If it's a new node, add its colour to the colour list.
            if node not in self.tree:
                self.color += [f"rgba{tuple(self.viridis[i + 1])}"]
                self.tree.add_edge(last, node)

            last = node

    def plot_network(
        self,
        peitsch_code: "int | str",
        min_size: int = 10,
        max_size: int = 50,
        to_percent: bool = True
    ) -> object:
        """Plot a network graph using plotly.

        Parameters
        ----------
        peitsch_code : int | str
            The peitsch code used to do the network.
        min_size : int, optional
            The minimal size of a plotted node, by default 10.
        max_size : int, optional
            The maximal size of a plotted node, by default 50.

        Returns
        -------
        object
            A plotly plot.
        """
        unorder_color: "list[int]" = []
        order_color: "list[int]" = []

        for m_i in tqdm(self.matrix_index, "  COMPUTING COLORATION"):
            if len(m_i) == 1:
                print(self.order_matrix[m_i, :][:, m_i])
                order_color += [-1]
                unorder_color += [-1]
            else:
                flat: np.ndarray = self.order_matrix[m_i, :][:, m_i].flatten()
                flat_sum = sum(sum(i) for i in zip(*flat[flat != 0]))

                length: int = sum(len(i) for i in zip(*flat[flat != 0]))
                to_mean: int = (length ** 2 - length) / 2

                if flat_sum / 2 / to_mean > 1:
                    print(flat_sum / 2 / to_mean)

                order_color += [flat_sum / 2 / to_mean]

                flat: np.ndarray = self.unorder_matrix[m_i, :][:, m_i].flatten(
                )
                flat_sum = sum(sum(i) for i in zip(*flat[flat != 0]))

                length: int = sum(len(i) for i in zip(*flat[flat != 0]))
                to_mean: int = (length ** 2 - length) / 2

                if flat_sum / 2 / to_mean > 1:
                    print(flat_sum / 2 / to_mean)

                unorder_color += [flat_sum / 2 / to_mean]

        order_color = np.array(order_color)

        cmap: object = colormaps["inferno_r"]
        v_order: np.ndarray = cmap(order_color)
        v_order *= np.array([255, 255, 255, 1])
        v_order = np.array(
            list(map(lambda line: f"rgba{tuple(line)}", v_order)))
        v_order[order_color == -1] = "rgba(150, 150, 150, 0.5)"
        v_order = list(v_order)

        unorder_color = np.array(unorder_color)

        cmap: object = colormaps["inferno_r"]
        v_unorder: np.ndarray = cmap(unorder_color)
        v_unorder *= np.array([255, 255, 255, 1])
        v_unorder = np.array(list(map(lambda line: f"rgba{tuple(line)}",
                                      v_unorder)))
        v_unorder[unorder_color == -1] = "rgba(150, 150, 150, 0.5)"
        v_unorder = list(v_unorder)

        max_size -= min_size

        if to_percent:
            # Percentage of the leaves size values.
            size: "list[float]" = list(np.array(self.size) /
                                       np.array(self.order_abs_size) * max_size
                                       + min_size)
        else:
            # Log the value to have a better view on the nodes' size diversity.
            log_size: np.ndarray = np.log(np.array(self.size) + 1)
            size: "list[float]" = list(np.array(log_size) / max(log_size)
                                       * max_size + min_size)

        # Craft the network.
        net_plot: object = igviz.plot(
            G=self.tree,
            title="",
            color_method=v_unorder,
            size_method=size,
            layout="kamada",
            colorscale=None,
            node_opacity=1,
            highlight_neighbours_on_hover=True,
            transparent_background=False,
            showlegend=True
        )

        # Modify traces attributs.
        net_plot.update_traces(
            marker_showscale=False,
            line_color="#DDD",
            line_width=1
        )

        # Modify the general layout.
        net_plot.update_layout(
            plot_bgcolor="white",
            legend_title=("<b>Network of SCOPe tree<br />classification for the"
                          f"<br />Peitsch code {peitsch_code}</b>"),
            font=dict(size=14),
            margin=dict(l=30, r=30, t=30, b=30),
            coloraxis_showscale=True
            # updatemenus=[
            #     dict(
            #         buttons=list([
            #             dict(
            #                 args=["marker", dict(color=v_unorder, size=size,
            #                       opacity=1)],
            #                 label="Bray-Curtis<br />distance",
            #                 method="restyle"
            #             ),
            #             dict(
            #                 args=["marker", dict(color=v_order, size=size,
            #                       opacity=1)],
            #                 label="O(NP)",
            #                 method="restyle"
            #             )
            #     ]),
            #     type="buttons",
            #     direction="right",
            #     pad={"r": 10, "t": 10},
            #     showactive=True,
            #     x=0,
            #     xanchor="left",
            #     y=1.47,
            #     yanchor="bottom",
            #     font_color="black"
            #     )
            # ]
        )

        # Modify legends' labels.
        net_plot["data"][0]["name"] = "Edges"
        net_plot["data"][1]["name"] = "Nodes"

        # Add the size in the hover text.
        order_hover: np.ndarray = np.array(order_color) * 100
        unorder_hover: np.ndarray = np.array(unorder_color) * 100
        percent_hover: np.ndarray = (np.array(size) - 10) / 4 * 10

        new_hover: "list[str]" = []

        for i, shift in enumerate(tqdm(self.size, "       MODIFYING HOVER")):
            if order_hover[i] < 0:
                order: str = f"Order context: None<br>"
            else:
                order: str = f"Order context: {order_hover[i]:.1f}<br>"

            if unorder_hover[i] < 0:
                unorder: str = f"Unorder context: None<br>"
            else:
                unorder: str = f"Unorder context: {unorder_hover[i]:.1f}<br>"

            new_hover += [f"Leaf number: {shift}<br>"
                          f"{unorder}{order}"
                          f"Relative size: {percent_hover[i]:.1f} %<br>"
                          f"{net_plot['data'][1]['hovertext'][i]}"]

        net_plot["data"][1]["hovertext"] = new_hover

        # To set x and y lim of the circle legends.
        x_pos: float = max(list(net_plot["data"][1]["x"])) * 1.25
        y_min: float = median(list(net_plot["data"][1]["y"])) * 1.2
        y_max: float = max(list(net_plot["data"][1]["y"]))
        y_pos: np.ndarray = np.linspace(y_min, y_max, 5)

        # Log the value to have a better view on the nodes' size diversity.
        circle_size: np.ndarray = np.linspace(min(size), max(size), 5)

        # Add the circle legend (node size explanation).
        side_plot: go.Scatter = go.Scatter(
            mode="markers",
            x=[x_pos] * 5,
            y=y_pos,
            marker=dict(color="#000", size=circle_size, opacity=1),
            hoverinfo="skip",
            name="Relative size to SCOPe<br />count"
        )

        # Add the circle legend (node size explanation).
        legend_plot: go.Scatter = go.Scatter(
            mode="markers",
            x=[None],
            y=[None],
            marker=dict(color="rgba(150, 150, 150, 0.5)", size=50, opacity=1),
            hoverinfo="skip",
            name="Context not computed"
        )

        ticks: "list[int]" = list(np.linspace(
            0,
            100,
            5
        ).astype(int))

        colorbar_plot: go.Scatter = go.Scatter(
            mode="markers",
            x=[None],
            y=[None],
            marker=dict(
                color="rgba(0, 0, 0, 0)",
                opacity=0,
                colorbar=dict(
                    title="Bray-Curtis<br />distance",
                    tickvals=ticks,
                    ticktext=ticks,
                    lenmode="pixels",
                    len=200,
                    yanchor="bottom",
                    y=0,
                ),
                colorscale="inferno_r",
                cmin=0,
                cmax=100
            ),
            name=""
        )

        net_plot.add_trace(side_plot)
        net_plot.add_trace(legend_plot)
        net_plot.add_trace(colorbar_plot)

        # To add the legend of the grey transparent nodes.
        for i, y in enumerate(y_pos):
            net_plot.add_annotation(
                xanchor="center",
                yanchor="middle",
                x=x_pos * 0.85,
                y=y,
                text=f"<b>{(circle_size[i] - 10) / 4 * 10:.0f}</b>",
                showarrow=False,
                font_color="#000",
                align="center"
            )

        net_plot.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            opacity=1,
            line=dict(color="rgba(0, 0, 0, 1)", width=1)
        )

        return net_plot


def get_domain(path: str, code: int) -> "tuple":
    """Get a domain list from a given Peitsch code.

    Parameters
    ----------
    path : str
        Path to pyHCA output file.
    code : int
        The Peitsch code.

    Returns
    -------
    list[str]
        The list of extracted domains.
    """
    last_power = 0
    cluster = ""

    # To translate a Peitsch code into a hydrophobic cluster.
    while code > 0:
        power = 0

        # When 2 ** power is greater than the code, power - 1 is the index where
        # we can put a 1 into the hydrophobic cluster list/text.
        while 2 ** power < code:
            power += 1

        # When we reach the last position, we don't have to do `-1` to the
        # power.
        if power != 0:
            power -= 1

        # Done only when the first `1` is add to the cluster.
        if cluster != "":
            cluster += "0" * (last_power - power - 1)

        last_power = power
        code -= 2 ** power
        cluster += "1"

    domain_list: "list[str]" = []
    context_dict: "dict[str : list]" = {}
    in_domain: bool = False
    domain: str = ""

    # Parse the file to check if a given cluster is in a domain.
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), "    PARSING pyHCA FILE"):
            # Only checking domain ID and cluster line.
            if line[0] != ">" and not line.startswith("cluster"):
                continue

            # Domaine line.
            if line[0] == ">":
                if in_domain:
                    context_dict[domain] = list(center_context(
                        context=np.array([context]),
                        window=10,
                        center=cluster,
                        add_gap=False,
                        keep_center=False
                    ))

                domain = "d" + line.split()[0][2:]
                context: "list[str]" = []
                in_domain = False
                continue

            hc: str = line.strip().split()[-1]
            context += [hc]

            # Cluster line: if we found the exact same cluster, add it to our
            # list.
            if cluster != hc:
                continue

            in_domain |= True
            domain_list += [domain]

    length: int = 0

    for content in context_dict.values():
        length += len(content)

    keys: "list[str]" = list(context_dict.keys())

    order_matrix: np.ndarray = np.zeros((length, length), dtype=float)
    unorder_matrix: np.ndarray = np.zeros((length, length), dtype=float)

    index_dict: "dict[str, int]" = {}
    domain_dict: "dict[str, int]" = {}

    index: int = 0

    for shift_i, i in enumerate(tqdm(keys[:-1], "    MATRIX COMPUTATION")):
        for j in keys[shift_i + 1:]:
            order: "list[float]" = []
            unorder: "list[float]" = []

            context: "list[str]" = context_dict[i] + context_dict[j]

            d_cont: "list[str]" = [i] * len(context_dict[i]) \
                + [j] * len(context_dict[j])
            n_cont: "list[int]" = list(range(len(context_dict[i]))) \
                + list(range(len(context_dict[j])))

            for pos_a, a in enumerate(context[:-1]):
                if len(a) <= 0:
                    continue

                dict_key_a: str = f"{d_cont[pos_a]}_{n_cont[pos_a]}"

                if dict_key_a not in index_dict:
                    index_dict[dict_key_a] = index
                    index += 1

                i_a: int = index_dict[dict_key_a]

                if d_cont[pos_a] not in domain_dict:
                    domain_dict[d_cont[pos_a]] = [i_a]
                else:
                    domain_dict[d_cont[pos_a]] += [i_a]

                for pos_b, b in enumerate(context[pos_a + 1:]):
                    if len(b) <= 0:
                        continue

                    pos_b += 1

                    dict_key_b: str = f"{d_cont[pos_b]}_{n_cont[pos_b]}"

                    if dict_key_b not in index_dict:
                        index_dict[dict_key_b] = index
                        index += 1

                    i_b: int = index_dict[dict_key_b]

                    if d_cont[pos_b] not in domain_dict:
                        domain_dict[d_cont[pos_b]] = [i_b]
                    else:
                        domain_dict[d_cont[pos_b]] += [i_b]

                    Context: object = PairewiseContextAnalyzer(a, b)
                    Context.compute_distance()

                    order = Context.distance[0]
                    unorder = Context.distance[1]

                    order_matrix[i_a][i_b] = order
                    order_matrix[i_b][i_a] = order
                    unorder_matrix[i_a][i_b] = unorder
                    unorder_matrix[i_b][i_a] = unorder

    # Return a list of unique elements, as far as a same Peitsch code can be
    # multiple time in the same domain.
    return list(set(domain_list)), [order_matrix, unorder_matrix, domain_dict]


if __name__ == "__main__":
    PEITSCH_CODE: int = [147, 201, 921]

    for code in PEITSCH_CODE:
        domain_list, data_list = get_domain(
            "data/pyHCA_SCOPe_30identity_globular.out",
            code
        )
        
        scope_tree: Scope = Scope(
            "data/SCOPe_2.08_classification.txt",
            order_matrix=data_list[0],
            unorder_matrix=data_list[1],
            index_dict=data_list[2]
        )

        for i, domain in enumerate(tqdm(domain_list, "   PARSING DOMAIN LIST")):
            scope_tree.add_domain(domain)

        plot: go.FigureWidget = scope_tree.plot_network(
            peitsch_code=code
        )

        for i, label in enumerate(plot["data"][1]["hovertext"]):
            node_name: str = label.split("<br>")[-2]

            if node_name[-1] == "0":
                continue
            if len(node_name) != 7:
                continue
            else:
                plot.add_annotation(
                    xanchor="center",
                    yanchor="middle",
                    x=plot["data"][1]["x"][i],
                    y=plot["data"][1]["y"][i],
                    text=f"<b> {node_name[-1]} </b>",
                    showarrow=False,
                    font_color="black",
                    align="center",
                    font=dict(size=14),
                    bgcolor="rgba(255, 255, 255, 0.6)"
                )

        plot.write_html(
            f"/home/lrouaud/Téléchargements/{code}_network.html",
            full_html=False,
            include_plotlyjs="cdn"
            # include_plotlyjs=("../../node_modules/plotly.js-dist-min/"
            #                   "plotly.min.js")
        )
