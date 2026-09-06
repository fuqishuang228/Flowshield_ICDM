import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR=Path('/workspace/Llama4AML_250721/explainability/sankey_parallel_updated')

OUTPUT_DIR = Path('/workspace/Llama4AML_250721/explainability/flow_sankey_test')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  

SOURCE_TX_DIR = Path('/workspace/Llama4AML_250721/explainability/source_tx')

external_flow_file = '/workspace/Llama4AML_250721/dataset/final_all_narrative_flow_with_label_tS_processed_new.csv'
external_df = pd.read_csv(external_flow_file, usecols=['index', 'transfer_type'])

def get_source_indices(index):
    source_file = SOURCE_TX_DIR / f"{index}_source_tx.csv"
    if not source_file.exists():
        return "unknown"

    try:

        with open(source_file, 'r') as f:
            header = f.readline()
            if 'index' not in header:
                return "unknown"

        df_src = pd.read_csv(source_file)
        if 'index' not in df_src.columns or df_src.empty:
            return "unknown"
        
        return ", ".join(str(i) for i in df_src['index'].dropna().astype(int).tolist())

    except Exception as e:
        return "unknown"



CSV_PATH = Path('/workspace/Llama4AML_250721/explainability/sankey_parallel_updated/21_sankey_augmented.csv')


try:
    csv_file = CSV_PATH
    file_parts = csv_file.stem.split("_")
    local_flow_id = file_parts[-1] if file_parts[-1].isdigit() else "unknown"

    df = pd.read_csv(csv_file)
    df = df.merge(external_df, on='index', how='left', suffixes=('', '_external'))

    df['source_indices'] = df['index'].apply(get_source_indices)


    df = df[df['value'] > 0]

    df = df[['from', 'to', 'value', 'tokenName', 'transaction_type', 'timeStamp', 'index','source_indices','transfer_type']].copy()




    def simplify(addr):
        return addr[:6]

    df['source_label'] = df['from'].apply(simplify)
    df['target_label'] = df['to'].apply(simplify)


    nodes = pd.Index(df['source_label'].tolist() + df['target_label'].tolist()).unique().tolist()
    node_dict = {name: i for i, name in enumerate(nodes)}
    df['source_id'] = df['source_label'].map(node_dict)
    df['target_id'] = df['target_label'].map(node_dict)

    df['time_str'] = pd.to_datetime(df['timeStamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')



    unique_tokens = df['tokenName'].dropna().unique()
    colors = plt.cm.tab20.colors  # 20 个颜色
    token_color_map = {
        token: f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.6)"
        for token, (r, g, b) in zip(unique_tokens, colors)
    }
    token_color_map['unknown'] = 'rgba(160,160,160,0.4)'


    df['edge_color'] = df['tokenName'].map(token_color_map).fillna(token_color_map['unknown'])

    df['hover_info'] = (
        'Tx Index: ' + df['index'].astype(str) +
        '<br>Tx Semantics: ' + df['transaction_type'].fillna('Unknown') +
        '<br>Tx Time: ' + df['time_str'].astype(str) +
        '<br>From: ' + df['source_label'] + ' → ' + df['target_label'] +
        '<br>Value: ' + df['value'].map('{:,.2f}'.format) + ' ' + df['tokenName'].fillna('Unknown') +
        '<br>Upstream Tx Index: ' + df['source_indices']+
        '<br>Fund flow Type: ' + df['transfer_type'].fillna('Unknown')
    )



    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="rgba(100, 100, 200, 0.6)"
        ),
        link=dict(
            source=df['source_id'],
            target=df['target_id'],
            value=df['value'],
            color=df['edge_color'],
            customdata=df['hover_info'],
            hovertemplate="%{customdata}<extra></extra>"
        )
    )])



    index_str = csv_file.stem.split("_")[0]

    fig.update_layout(
        title_text=f"Subflow {local_flow_id} on Day 1",
        font_size=16,
        width=1400,
        height=600
    )


    html_file = OUTPUT_DIR / f"{csv_file.stem}.html"
    fig.write_html(str(html_file))


except Exception as e:
    print(f"❌  {csv_file.name} ：{e}")
