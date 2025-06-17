import os
import re
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def get_json(fn):
    j = None
    try:
        with open(fn, 'r') as f:
            j = json.load(f)
    except Exception as e:
        print(f"Opening file {fn} failed with exception {e}")
        #sys.exit(1)
    return j

def get_result_files(directory):
    all_files = defaultdict(lambda: defaultdict(list))
    print(f"Getting result files from {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            #all_files.append(os.path.join(root, file))
            r = root[len(directory)+1:].split('/')

            if len(r) < 2:
                continue
            all_files[r[0]][r[1]].append(os.path.join(root, file))
            #all_files[(r[0], r[1])].append(file)
    return all_files

def get_highest_concurrency_file(files, sep):
    t = [(int(f[f.rfind('C')+1:f.rfind(sep)]), f) for f in files]
    return max(t)

def find_model_config(model_configs, model):
    m = model.replace('--', '/')
    for mc in model_configs:
        if m == mc['model']:
            return mc

def extract_results_from_json(fn):
    j = get_json(fn)
    return {
        'Model': j['model_id'],
        'Input Tokens': int(j['total_input_tokens']/j['num_prompts']),
        'Output Tokens': int(j['total_output_tokens']/j['num_prompts']),
        'Concurrency': int(j['request_rate']),
        'Output tput (tokens/sec)': round(j['output_throughput'], 2),
        'Total tput (tokens/sec)': round(j['total_token_throughput'], 2),
        'TTFT avg (ms)': round(j['mean_ttft_ms'], 2),
        'TTFT P50 (ms)': round(j['p50_ttft_ms'], 2),
        'TTFT P90 (ms)': round(j['p90_ttft_ms'], 2),
        'TTFT P99 (ms)': round(j['p99_ttft_ms'], 2),
        'TPOT avg (ms)': round(j['mean_tpot_ms'], 2),
        'TPOT P50 (ms)': round(j['p50_tpot_ms'], 2),
        'TPOT P90 (ms)': round(j['p90_tpot_ms'], 2),
        'TPOT P99 (ms)': round(j['p99_tpot_ms'], 2),
        'Request Throughput': round(j['request_throughput'], 2),
        'Request Latency avg (s)': round(j['mean_e2el_ms']/1000, 2),
        'Request Latency P50 (s)': round(j['p50_e2el_ms']/1000, 2),
        'Request Latency P90 (s)': round(j['p90_e2el_ms']/1000, 2),
        'Request Latency P99 (s)': round(j['p99_e2el_ms']/1000, 2),
    }

def extract_results_from_json_embed(model, fns, model_config):
    j = get_json(fns)
    mc = model_config['test_parameters']

    result = {
            'Model': model,
            'Input Tokens': int(j['total_input_tokens']/j['num_prompts']),
            'Concurrency': int(j['request_rate']),
            'Embedding dimension': mc["embedding_dimension"],
            'Num parameters': mc["num_parameters"],
            'Max tokens': mc["max_tokens"],
            'MTEB rank': mc["mteb_rank"],
            'Memory (MB)': mc["mem_usage(MB)"],
            'Request Latency avg (ms)': round(j['mean_e2el_ms'], 2),
            'Request Latency P50 (ms)': round(j['p50_e2el_ms'], 2),
            'Request Latency P90 (ms)': round(j['p90_e2el_ms'], 2),
            'Request Latency P99 (ms)': round(j['p99_e2el_ms'], 2),
            'Num Requests': int(j['num_prompts']),
            'Throughput (req/sec)': round(j['request_throughput'], 2)
    }
    return result



def parse_num_parameters(val):
    if isinstance(val, str):
        if val.endswith('B'):
            return float(val[:-1]) * 1000  # 1B -> 1000M
        if val.endswith('M'):
            return float(val[:-1])
    return float(val)



def param_group(num_m):
    if num_m >= 7000:
        return '7B+'
    elif num_m >= 1000:
        return '1B-7B'
    elif num_m >= 300:
        return '300M-1B'
    else:
        return '<300M'

def shorten_model_name(name):
    # Split by '--', '/', or '-' and take the last part
    if '--' in name:
        return name.split('--')[-1]
    elif '/' in name:
        return name.split('/')[-1]
    elif '-' in name:
        return name.split('-')[-1]
    else:
        return name

metric_map = { "latency": 'Request Latency P90 (ms)',
               "throughput": 'Throughput (req/sec)'}


def generate_chart(df, result_dir, scale='linear', metric_key='latency'):


    if metric_key in metric_map:
        metric = metric_map[metric_key]
    else:
        print(f"Invalid metric key: {metric_key}. Valid keys are: {list(metric_map.keys())}")
        return
    tag = os.path.basename(result_dir.strip('/')) + "_" + metric_key
    print(tag)
    base_dir = os.path.dirname(result_dir)
    graph_file = os.path.join(base_dir, tag+'_graph.png')

    df['Param Group'] = df['Num parameters (M)'].apply(param_group)
    df['Model Short'] = df['Model'].apply(shorten_model_name)

    for tokens in [128, 256, 512]:
        df_subset = df[df['Input Tokens'] == tokens]
        plt.figure(figsize=(12, 6))
        for group in ['7B+', '1B-7B', '300M-1B', '<300M']:
            group_df = df_subset[df_subset['Param Group'] == group]
            bars = plt.bar(group_df['Model'], group_df[metric], label=group)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height,
                         f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        # Dynamically adjust y-axis limits to add padding for text
        #max_height = max([bar.get_height() for bar in bars]) if bars else 0
        #plt.ylim(0, max_height * 1.2)  # Add 20% padding above the tallest bar


        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.title(f'{metric} for Input Tokens = {tokens}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Param Group')
        plt.yscale(scale)  # Add this line for logarithmic y-axis
        plt.tight_layout()
        plt.savefig(graph_file.replace('.png', f'_{tokens}_{scale}.png'))
        plt.close()

def generate_comparison_chart(df1, df2, df1_file, df2_file, scale='linear', metric_key='latency'):
    if metric_key in metric_map:
        metric = metric_map[metric_key]
    else:
        print(f"Invalid metric key: {metric_key}. Valid keys are: {list(metric_map.keys())}")
        return
    dir_name = os.path.dirname(df1_file)
    d1_base = os.path.basename(df1_file)
    d2_base = os.path.basename(df2_file)

    #system1 = os.path.basename(os.path.dirname(df1_file))
    #system2 = os.path.basename(os.path.dirname(df2_file))
    #print(system1)
    #print(system2)
    if 'gnr' in df1_file:
        if 'spr' in df2_file:
            d1_label = 'gnr'
            d2_label = 'spr'
    elif 'spr' in df1_file:
        if 'gnr' in df2_file:
            d1_label = 'spr'
            d2_label = 'gnr'
    else:
        d1_label = d1_base
        d2_label = d2_base

    if d1_label != d1_base:
        graph_file = os.path.join(dir_name, f'{d1_label}_{d1_base}&{d2_label}_{d2_base}_{metric_key}_comparison.png')
    else:
        graph_file = os.path.join(dir_name, f'{d1_base}&{d2_base}_{metric_key}_comparison.png')
    # Define a function to shorten model names


    df1['Param Group'] = df1['Num parameters (M)'].apply(param_group)
    df2['Param Group'] = df2['Num parameters (M)'].apply(param_group)
    df1['Model Short'] = df1['Model'].apply(shorten_model_name)
    df2['Model Short'] = df2['Model'].apply(shorten_model_name)



    for tokens in [128, 256, 512]:
        for group in ['7B+', '1B-7B', '300M-1B', '<300M']:
            df1_subset = df1[(df1['Input Tokens'] == tokens) & (df1['Param Group'] == group)]
            df2_subset = df2[(df2['Input Tokens'] == tokens) & (df2['Param Group'] == group)]

            # Merge on Model Short to align bars
            merged = pd.merge(
                df1_subset[['Model Short', metric, 'Num parameters (M)']],
                df2_subset[['Model Short', metric, 'Num parameters (M)']],
                on='Model Short', how='outer', suffixes=('_df1', '_df2')
            ).fillna(0)

            # Combine Model Short and Num parameters (M) for x-axis labels
            merged['Bar Label'] = merged['Model Short']
            #merged['Bar Label'] = merged['Model Short'] + " (" + merged['Num parameters (M)'].astype(int).astype(str) + "M)"

            x = np.arange(len(merged))
            width = 0.35

            fig, ax = plt.subplots(figsize=(14, 6))
            bars1 = ax.bar(x - width/2, merged[f'{metric}_df1'], width, label=d1_label)
            bars2 = ax.bar(x + width/2, merged[f'{metric}_df2'], width, label=d2_label)

            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                diff = height2 / height1
                ax.text(bar1.get_x() + bar1.get_width() / 2, height1,
                       f'{height1:.1f} ms', ha='center', va='bottom', fontsize=8)
                ax.text(bar2.get_x() + bar2.get_width() / 2, height2,
                       f'{height2:.1f} ms', ha='center', va='bottom', fontsize=8)
                diff_color = 'green' if diff > 1 else 'red'
                ax.text(bar1.get_x() + bar1.get_width(), max(height1, height2) + 2,
                       f'{diff:.2f}', ha='center', va='bottom', fontsize=10, color=diff_color)

            # Adjust y-axis limits to add padding for text
            max_height = max([bar.get_height() for bar in bars1 + bars2])
            # Add a note to the chart
            #ax.text(0.5, max_height * 1.1, f'Note: Latency values, smaller is better',
            #       ha='center', va='top', fontsize=12, color='blue', transform=ax.transAxes)

            ax.set_ylim(0, max_height * 1.2)  # Add 20% padding above the tallest bar

            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f"{group} - {metric} for Input Tokens = {tokens}")
            ax.set_xticks(x)
            ax.set_xticklabels(merged['Bar Label'], rotation=45, ha='right')
            ax.legend()
            ax.set_yscale(scale)
            plt.tight_layout()
            plt.savefig(graph_file.replace('.png', f'_{tokens}_{group}_{scale}.png'))
            plt.close()

def process_result_directory(args):
    result_dir = args.result_dir
    print(result_dir)
    tag = os.path.basename(result_dir.strip('/'))
    print(tag)
    base_dir = os.path.dirname(result_dir)
    csv_file = os.path.join(base_dir, tag+'.csv')
    small_csv = os.path.join(base_dir, tag+'_small.csv')

    files = get_result_files(result_dir)

    confd = os.path.join(os.getcwd(), f'configs/{args.p}/')
    model_config_file = os.path.join(confd, 'models_embed.json')
    if not os.path.exists(model_config_file):
        print(f"Model config file {model_config_file} does not exist")
        return

    model_config = get_json(model_config_file)
    sep = args.sep
    if args.test_type == 'benchmark':
        sep = '-'
    elif args.test_type == 'sweep':
        sep = '.'

    values = []
    for model, batches in files.items():
        mc = find_model_config(model_config, model)

        for fns in batches.values():
            if args.test_type == 'sweep':
                _, fn = get_highest_concurrency_file(fns, sep)
            else:
                fn = fns[0]
            values.append(extract_results_from_json_embed(model, fn, mc))
    df = pd.DataFrame(values)
    df['Num parameters (M)'] = df['Num parameters'].apply(parse_num_parameters)
    df = df.sort_values(by=['Input Tokens', 'Num parameters (M)',   'Model', 'Concurrency'], ascending=[True, False, True, True])
    df.to_csv(csv_file, index=False)
    df[['Model', 'Input Tokens', 'Concurrency', 'Throughput (req/sec)', 'Request Latency P90 (ms)']].to_csv(small_csv, index=False)
    print(f"Results saved to {csv_file}")
    print(f"Small results saved to {small_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate result csv file from tester results directory")
    parser.add_argument("-d", "--result_dir", type=str, help="result dir path", required=True)
    parser.add_argument("-p", type=str, help="test platform", required=True)
    parser.add_argument("-g", "--generate_graph", action="store_true", help="output charts")
    parser.add_argument("-cg", "--comparison-graph-file", action="store_true", help="output comparison charts")
    parser.add_argument("-d1", "--result-csv", type=str, help="path to a generated \
                        csv file for the ist set of results if require to generate a comparison graph", default=None)
    parser.add_argument("-d2", "--second-result-csv", type=str, help="path to a generated \
                        csv file for the 2nd set of results if require to generate a comparison graph", default=None)
    parser.add_argument("-tt", "--test-type", type=str, help="type of tests of the result files", default='benchmark')
    parser.add_argument("-s", "--scale", type=str, help="output graph scale type [log|linear]", default='linear')
    parser.add_argument("-e", "--sep", type=str, help="separator for the file name", default='-')
    args = parser.parse_args()


    if args.result_dir is None:
        print("Please provide a result directory")
        return

    if args.result_dir and args.result_csv:
        print(f"NOTE: Both result_dir and d1 are provided, will use d1 instead of JSON files in result_dir!!!\n \
                Please make sure d1 is a valid csv file and is generated from the data in {args.result_dir}")
        df1 = pd.read_csv(args.result_csv)
    else:
        df1 = process_result_directory(args)

    #if args.generate_graph:
    #    generate_chart(df1, args.result_dir, scale=args.scale, metric_key='latency')
    #    generate_chart(df1, args.result_dir, scale=args.scale, metric_key='throughput')
    #    print(f"Graph saved to {args.result_dir}")

    if args.comparison_graph_file:
        if args.second_result_csv is None:
            print("Please provide the 2nd result csv files for comparison")
            return
        df2 = pd.read_csv(args.second_result_csv)
        df1_file = os.path.basename(args.result_csv) if args.result_csv else args.result_dir
        generate_comparison_chart(df1, df2, df1_file, args.second_result_csv,
                                   scale=args.scale, metric_key='latency')
        # Throughput in quick test is not useful, so not generating the chart
        #generate_comparison_chart(df1, df2, df1_file, args.second_result_csv,
        #                           scale=args.scale, metric_key='throughput')

    print("Done")

if __name__ == "__main__":
    main()
