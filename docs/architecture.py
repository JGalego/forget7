"""Architecture diagram as code. Regenerate the PNG with:

    uv run --with diagrams --with pillow python docs/architecture.py

Needs Graphviz. Layout notes, because dot is easy to tangle:
  * Columns, left to right: your machine (0-1), then AWS (2-4). Every edge points
    rightwards in rank order; arrows that should *look* leftwards are drawn with
    dir=back (`back()` below) so they don't fight the ranking.
  * Rows are pinned with rank=same groups and flat edges (newrank=true). In an LR
    layout a flat edge puts its tail *below* its head. These are hints, not hard
    constraints: this order is the one dot's crossing minimisation also agrees with.
  * Nodes are fixed at 1.4in wide, so node labels stay short (one idea per line);
    the API names live on edge labels, which dot sizes to fit.
  * The execution role is a raw HTML-table node around a tightly cropped hat icon,
    with a port on the icon cell. Edges attach to the hat itself (not to the node's
    padded box), and dot uses the port offset when it straightens the PassRole edge.
"""

import tempfile
from pathlib import Path

import diagrams
from diagrams import Cluster, Diagram, Edge
from diagrams.aws.general import User
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.security import IAMRole
from diagrams.aws.storage import S3
from diagrams.onprem.client import Client
from diagrams.programming.language import Python
from PIL import Image

OUT = Path(__file__).with_name("architecture")
GRAPH = {
    "pad": "0.4",
    "nodesep": "0.7",
    "ranksep": "1.6",
    "splines": "spline",
    "fontsize": "13",
    "newrank": "true",
    "compound": "true",
}
NODE = {"fontsize": "12"}
CLUSTER = {"fontsize": "15", "margin": "28", "labeljust": "l"}
SUB = {"fontsize": "13", "margin": "20", "labeljust": "l"}
RED = "firebrick"
EDGE_FONT = {"fontname": "Sans-Serif", "fontsize": "13", "fontcolor": "#2D3436"}  # what diagrams.Edge uses
# Weight for every in-row edge. dot packs each cluster at minimum node spacing and centres the
# shorter cluster against the taller one, which tilts cross-cluster rows; at 100 that still wins,
# at 1000 the rows win. It has to be on every row edge, or the unweighted rows bend instead.
ROW = "1000"


def back(tail, head, **attrs):
    """Draw head -> tail while ranking tail before head (keeps dot's columns intact)."""
    tail.connect(head, Edge(tail, reverse=True, **attrs))


def same_column(diag, *ids):
    diag.dot.body.append("{rank=same; " + " ".join(f'"{i}"' for i in ids) + "}")


def html_icon_node(cluster, node_id, icon_cls, text, width_pt=100.8):
    """A node whose icon cell is cropped to the glyph, so ports sit on the glyph's edges."""
    src = Path(diagrams.__file__).parent.parent / icon_cls._icon_dir / icon_cls._icon
    img = Image.open(src)
    img = img.crop(img.getchannel("A").getbbox())
    tight = Path(tempfile.mkdtemp()) / src.name
    img.save(tight)
    height_pt = width_pt * img.height / img.width
    label = (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD PORT="icon" FIXEDSIZE="TRUE" WIDTH="{width_pt:.0f}" HEIGHT="{height_pt:.0f}">'
        f'<IMG SRC="{tight}" SCALE="TRUE"/></TD></TR>'
        f'<TR><TD HEIGHT="26" VALIGN="BOTTOM">{text}</TD></TR></TABLE>>'
    )
    cluster.node(node_id, label, shape="plain", fixedsize="false", style="", image="")


with Diagram(
    "",
    filename=str(OUT),
    outformat="png",
    show=False,
    direction="LR",
    curvestyle="curved",
    graph_attr=GRAPH,
    node_attr=NODE,
) as diag:
    with Cluster("Your machine", graph_attr=CLUSTER):
        you = User("you\naws login")
        data = Python("data.py\nforget + retain")
        files = Client("out/\nlogs, charts")
        loop = Python("train.py\nthe loop")
        analysis = Python("bisect.py\nheatmap.py")

    with Cluster("AWS Region (us-west-2)", graph_attr=CLUSTER) as aws:
        html_icon_node(aws, "role", IAMRole, "execution role")
        with Cluster("Serverless GPUs", graph_attr=SUB):
            session = SagemakerTrainingJob("Training Session\ngpt-oss-20b")
            sampler = Sagemaker("sampling\nsession")
        with Cluster("Checkpoints", graph_attr=SUB):
            registry = SagemakerModel("model package\ngroup")
            bucket = S3("S3 artifacts")

    # rows, top to bottom: analysis, the training loop, identity
    same_column(diag, you._id, data._id, files._id)
    you >> Edge(style="invis") >> data
    data >> Edge(style="invis") >> files
    same_column(diag, "role", session._id, sampler._id)  # role -> session order comes from "assumed by"
    session >> Edge(style="invis") >> sampler

    # your machine
    data >> Edge(label="Datum objects\nweights < 0 on 7", weight=ROW) >> loop
    back(files, loop, label="logs", style="dotted")
    back(files, analysis, label="charts", style="dotted", weight=ROW)

    # identity: you hand the role to the service, the session assumes it
    diag.dot.edge(you._id, "role:icon", label="iam:PassRole", style="dashed", minlen="2", weight=ROW, **EDGE_FONT)
    # the table's top border is the hat's top, so this leaves the hat itself; it ends at the
    # cluster border because every session (training or sampling) runs as this role.
    # (No port on this end: graphviz 2.43 aborts on lhead + a tail port.)
    # weight 0: a flat edge, so it only sets the order and must not pull the role up out of the you row
    diag.dot.edge(
        "role",
        session._id,
        xlabel="assumed by   ",
        style="dashed",
        weight="0",
        lhead="cluster_Serverless GPUs",
        **EDGE_FONT,
    )

    # the loop: client-side Python, GPU-side primitives
    (
        loop
        >> Edge(
            label="forward_backward\noptim_step\nsample, forward",
            color=RED,
            penwidth="2",
            forward=True,
            reverse=True,
            weight=ROW,
        )
        >> session
    )
    analysis >> Edge(label="compute_logprobs\nsample", forward=True, reverse=True, weight=ROW) >> sampler

    # checkpoints
    session >> Edge(label="save_state\nsave_weights_for_sampler") >> registry
    registry >> bucket
    back(sampler, registry, label="model_package_arn", style="dashed")
