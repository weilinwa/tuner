import sys
import os
import shutil
import json
import logging
import subprocess
import shlex
import time
import requests
import argparse


MODEL_DIR_BASE = os.getcwd()+"/models"
TRACE_DIR_BASE = os.getcwd()+"/traces_server"
NGINX_DIR_BASE = os.getcwd()+"/nginx"
RESULTS_DIR_BASE = os.getcwd()+"/results/result_"+str(int(time.time()))
SERVED_MODEL_NAME = "model_in_test"
HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN', "")
PROXY_ENV = f"-e HTTP_PROXY={os.environ.get('HTTP_PROXY', '')} -e HTTPS_PROXY={os.environ.get('HTTPS_PROXY', '')} -e NO_PROXY={os.environ.get('NO_PROXY', '')}"
BENCHMARK_DIR_BASE="/home/weilinwa/AI/aws/vllm_2"
BENCHMARK_DIR_SCRIPTS_BASE=os.getcwd()

os.makedirs(RESULTS_DIR_BASE)
logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime}.{msecs:.0f} - {levelname} - {message}",
        style="{",
        datefmt="%H:%M:%S",
        handlers = [
                logging.FileHandler(RESULTS_DIR_BASE + '/tuner.log'),
                logging.StreamHandler()
            ]
        )


def run_docker_cmd(docker_command, is_exit=True):
    logging.debug(f"Running docker command: {docker_command}")
    try:
        process = subprocess.Popen(shlex.split(docker_command), shell=False)
        process.wait()
        #result = subprocess.run("ls", check=True, capture_output=True, text=True)
    except Exception as e:
        logging.exception(f"Error running docker cmd, exception encountered is {e}")
        if is_exit:
            sys.exit(1)
    #return result


def stop_nginx():
    docker_command1 = "docker stop nginx-lb"
    docker_command2 = "docker rm nginx-lb"
    run_docker_cmd(docker_command1)
    run_docker_cmd(docker_command2, is_exit=False)


def stop_vllm(numa_conf):
    stop_nginx()

    for i in range(len(numa_conf)):
        container_name = 'vllm'+str(i)
        docker_command1 = f"docker stop {container_name}"
        docker_command2 = f"docker rm {container_name}"
        logging.debug(f"Stopping and removing container {container_name}")
        run_docker_cmd(docker_command1)
        run_docker_cmd(docker_command2, is_exit=False)

    logging.info("Waiting for 15s after stopping and removing vllm containers")
    time.sleep(15)


def get_model_res_dir(model):
    m = model.replace('/', '--')
    return f"{RESULTS_DIR_BASE}/{m}"


def run_benchmark(model, token_comb, containers_conf, qpc, is_warmup, it=1):
    container_image = containers_conf['benchmark']['image']
    cpus = containers_conf['benchmark']['cpuset']
    concurrency = token_comb['concurrency']
    inp_tokens = token_comb['inp_tokens']
    op_tokens = token_comb['op_tokens']
    served_model_name = SERVED_MODEL_NAME
    num_prompts = concurrency * qpc
    model_dir = f"{MODEL_DIR_BASE}"
    results_dir = get_model_res_dir(model) + f"/I{inp_tokens}-O{op_tokens}"
    if not os.path.exists(results_dir) and not is_warmup:
        os.makedirs(results_dir)
    results_file = f"result-C{concurrency}-iter{it}.json"
    results_file_container = f"/results/{results_file}"
    results_file_host = f"{results_dir}/{results_file}"

    docker_command = ""

    if is_warmup:
        docker_command = f"docker run -it --cpuset-cpus={cpus} --rm --net=host {PROXY_ENV} -v {model_dir}:/root/.cache -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} --entrypoint=python3 {container_image} /workspace/vllm/benchmarks/benchmark_serving.py --port 8000 --dataset-name random --request-rate {concurrency} --num-prompts {num_prompts} --random-input-len {inp_tokens} --random-output-len {op_tokens} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --served-model-name {served_model_name} --metric-percentiles 50,90,99 --max-concurrency {concurrency} --model {model}"
    else:
        docker_command = f"docker run -it --cpuset-cpus={cpus} --rm --net=host {PROXY_ENV} -v {model_dir}:/root/.cache -v {results_dir}:/results -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} --entrypoint=python3 {container_image} /workspace/vllm/benchmarks/benchmark_serving.py --port 8000 --dataset-name random --request-rate {concurrency} --num-prompts {num_prompts} --random-input-len {inp_tokens} --random-output-len {op_tokens} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --served-model-name {served_model_name} --metric-percentiles 50,90,99 --max-concurrency {concurrency} --save-result --result-filename {results_file_container} --model {model}"

    run_docker_cmd(docker_command)
    return results_file_host

def run_benchmark_embed(model, token_comb, containers_conf, qpc, is_warmup, it=1, trace=False):
    container_image = containers_conf['benchmark']['image']
    cpus = containers_conf['benchmark']['cpuset']
    concurrency = token_comb['concurrency']
    inp_tokens = token_comb['inp_tokens']
    served_model_name = SERVED_MODEL_NAME
    num_prompts = concurrency * qpc
    model_dir = f"{MODEL_DIR_BASE}"
    results_dir = get_model_res_dir(model) + f"/I{inp_tokens}"
    TRACE_ENV = "--profile" if trace else "" # TODO: trace is only applied on warmup run
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = f"result-C{concurrency}-iter{it}.json"
    results_file_container = f"/results/{results_file}"
    results_file_host = f"{results_dir}/{results_file}"

    docker_command = ""

    if is_warmup:
        docker_command = f"docker run -it --cpuset-cpus={cpus} --rm --net=host {PROXY_ENV} -v {BENCHMARK_DIR_BASE}/benchmarks/:/workspace/vllm/benchmarks/ -v {BENCHMARK_DIR_SCRIPTS_BASE}/benchmark_serving_embedding.py:/workspace/vllm/benchmarks/benchmark_serving_embedding.py -v {BENCHMARK_DIR_SCRIPTS_BASE}/backend_request_func.py:/workspace/vllm/benchmarks/backend_request_func.py -v {model_dir}:/root/.cache -v /home/weilinwa/AI/aws/test_scripts/tuner/traces/:/root/traces/ -e VLLM_TROCH_PROFILER_DIR=/root/traces/ -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} --entrypoint=python3 {container_image} /workspace/vllm/benchmarks/benchmark_serving_embedding.py --port 8000 --backend vllm-embed --endpoint /v1/embeddings --dataset-name random --request-rate {concurrency} --num-prompts {num_prompts} --random-input-len {inp_tokens} --random-output-len {inp_tokens} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --served-model-name {served_model_name} --metric-percentiles 50,90,99 --max-concurrency {concurrency} {TRACE_ENV} --model {model}"
    else:
        docker_command = f"docker run -it --cpuset-cpus={cpus} --rm --net=host {PROXY_ENV} -v {BENCHMARK_DIR_BASE}/benchmarks/:/workspace/vllm/benchmarks/ -v {BENCHMARK_DIR_SCRIPTS_BASE}/benchmark_serving_embedding.py:/workspace/vllm/benchmarks/benchmark_serving_embedding.py -v {BENCHMARK_DIR_SCRIPTS_BASE}/backend_request_func.py:/workspace/vllm/benchmarks/backend_request_func.py -v {model_dir}:/root/.cache -v {results_dir}:/results -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} --entrypoint=python3 {container_image} /workspace/vllm/benchmarks/benchmark_serving_embedding.py --port 8000 --backend vllm-embed --endpoint /v1/embeddings --dataset-name random --request-rate {concurrency} --num-prompts {num_prompts} --random-input-len {inp_tokens} --random-output-len {inp_tokens} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --served-model-name {served_model_name} --metric-percentiles 50,90,99 --max-concurrency {concurrency} --save-result --result-filename {results_file_container} --model {model}"

    run_docker_cmd(docker_command)
    return results_file_host


def run_benchmark_iters(model, token_comb, containers_conf, qpc, iterations):
    result_files = []
    for i in range(iterations):
        logging.info(f"Starting benchmark for - concurrency: {token_comb['concurrency']}, qpc: {qpc}, iteration: {i+1}")
        result_files.append(run_benchmark(model, token_comb, containers_conf, qpc, False, it=i+1))

    return result_files

def run_benchmark_iters_embed(model, token_comb, containers_conf, qpc, iterations):
    result_files = []
    for i in range(iterations):
        logging.info(f"Starting benchmark for - concurrency: {token_comb['concurrency']}, qpc: {qpc}, iteration: {i+1}")
        result_files.append(run_benchmark_embed(model, token_comb, containers_conf, qpc, False, it=i+1))

    return result_files


def launch_nginx(containers_conf):
    container_image = containers_conf['nginx']['image']
    cpus = containers_conf['nginx']['cpuset']
    logging.info("Launching nginx...")
    docker_command = f"docker run --rm -itd --cpuset-cpus={cpus} -p 8000:80 --network vllm_nginx -v {NGINX_DIR_BASE}/nginx_conf/:/etc/nginx/conf.d/ --name nginx-lb {container_image}"
    run_docker_cmd(docker_command, is_exit=False)
    time.sleep(2)


def launch_vllm(test, numa_conf, containers_conf, embed=False, gpu=False, trace=False):
    for i, n in enumerate(numa_conf):
        node = i
        cpuset = n['cpubind']
        mem =  n['membind']
        model_dir = f"{MODEL_DIR_BASE}"
        container_image = containers_conf['vllm']['image']
        dtype = test['dtype']
        served_model_name = SERVED_MODEL_NAME
        model = test['model']
        port = 8000 + node + 1 if len(numa_conf) > 1 else 8000
        container_name = f"vllm{node}"
        kv_cache = test['test_parameters']['kv_cache']
        node_cpus = n['node']
        memory_opt = f"-m {n['memory']}" if 'memory' in n else ""
        compile_config = 3
        OMP_ENV = "-e KMP_BLOCKTIME=1 -e KMP_TPAUSE=0 -e KMP_SETTINGS=0 -e KMP_FORKJOIN_BARRIER_PATTERN=dist,dist -e KMP_PLAIN_BARRIER_PATTERN=dist,dist "
        OMP_ENV += f"-e KMP_REDUCTION_BARRIER_PATTERN=dist,dist -e VLLM_V1_USE=1 -e VLLM_CPU_OMP_THREADS_BIND={cpuset}"
        trace_dir=f"{TRACE_DIR_BASE}"
        TRACE_ENV = f"-v {trace_dir}:/root/traces/ -e VLLM_TORCH_PROFILER_DIR=/root/traces/" if trace else ""
        #LOG_OPT=f"--log-driver json-file --log-opt max-size=10m --log-opt max-file=3"
        LOG_OPT = ''

#        docker_command = f"docker run -d --rm {PROXY_ENV} -p {port}:8000 --cpuset-cpus={cpuset} --cpuset-mems={mem} -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} -e VLLM_CPU_KVCACHE_SPACE={kv_cache} -v {model_dir}:/root/.cache --name {container_name} --ipc=host {container_image} --trust-remote-code --device cpu --dtype {dtype} --tensor-parallel-size 1 --enforce-eager --served-model-name {served_model_name} --model {model}"
        if gpu:
            # TODO: keep the OMP_ENV for GPU tests?
            docker_command = f"docker run --runtime nvidia --gpus all -d --rm --privileged=True {PROXY_ENV} -p {port}:8000 --network vllm_nginx --cpuset-cpus={node_cpus} --cpuset-mems={mem} {OMP_ENV} {LOG_OPT}  -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} -e VLLM_CPU_KVCACHE_SPACE={kv_cache} -v {model_dir}:/root/.cache {TRACE_ENV} --name {container_name} --ipc=host {container_image} --trust-remote-code --dtype {dtype} --tensor-parallel-size 1 --served-model-name {served_model_name} --model {model}"
        else:
            docker_command = f"docker run -d --rm --privileged=True {PROXY_ENV} -p {port}:8000 --network vllm_nginx --cpuset-cpus={node_cpus} --cpuset-mems={mem} {memory_opt} {OMP_ENV} {LOG_OPT} -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} -e VLLM_CPU_KVCACHE_SPACE={kv_cache} -v {model_dir}:/root/.cache {TRACE_ENV} --name {container_name} --ipc=host {container_image} --trust-remote-code --device cpu --dtype {dtype} --tensor-parallel-size 1 --served-model-name {served_model_name} --model {model} -O{compile_config}"

        run_docker_cmd(docker_command)
    logging.info("Waiting 60s for all VLLM containers to initialize")
    time.sleep(60)

    for i, n in enumerate(numa_conf):
        ready = False
        port = 8000 + i + 1 if len(numa_conf) > 1 else 8000
        for _ in range(75):
            try:
                response = requests.get(f"http://localhost:{port}/version", timeout=2)
                if response.status_code == 200:
                    logging.info("VLLM endpoints are available")
                    ready = True
                    break
            except (requests.ConnectionError, requests.Timeout) as  e:
                logging.info("VLLM not yet initialized, retrying in 60 seconds")

            time.sleep(60)
        if not ready:
            logging.info("Exiting test, VLLM endpoints are not available. Check vllm0 container llogs")
            sys.exit(1)

    launch_nginx(containers_conf)

    #Warmup run
    if embed:
        run_benchmark_embed(test['model'], {'inp_tokens': 128, 'op_tokens': 128, 'concurrency': 10}, containers_conf, 1, True, trace=trace)
    else:
        run_benchmark(test['model'], {'inp_tokens': 128, 'op_tokens': 128, 'concurrency': 2}, containers_conf, 1, True)


def prepare_tests(models_conf):
    tests = []
    #os.makedirs(RESULTS_DIR_BASE)
    for m in models_conf:
        results_dir = get_model_res_dir(m['model'])
        os.makedirs(results_dir)
        e = {'model': m['model'],
                'dtype': m['dtype'],
                'test_parameters': m['test_parameters']}
        tests.append(e)

    return tests


def prepare_single_test(model, params):
    tp = json.loads(params)
    tests = []
    results_dir = get_model_res_dir(model)
    os.makedirs(results_dir)
    e = {'model': model,
                'dtype': tp['dtype'],
                'test_parameters': tp['test_parameters']}
    tests.append(e)
    return tests


def run_download(container_image):
    #create dir for models download
    pwd = os.getcwd()
    model_dir = MODEL_DIR_BASE

    logging.info(f"Downloading models in {model_dir}...")
    #if os.path.exists(model_dir):
    #    shutil.rmtree(model_dir)
    #os.makedirs(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    docker_command = f"docker run --rm {PROXY_ENV} -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} -v {pwd}/configs:/workspace/configs -v {model_dir}:/root/.cache {container_image}"
    run_docker_cmd(docker_command, is_exit=False)


def get_json(fn, fail_on_empty=True):
    j = None
    try:
        with open(fn, 'r') as f:
            j = json.load(f)
    except Exception as e:
        logging.error(f"Opening file {fn} failed with exception {e}", exc_info=True)
        if not fail_on_empty:
            return {"error": str(e)}
        else:
            sys.exit(1)
    return j or {}


def get_configs(args):
    conf = {}
    global PROXY_ENV
    numa_fn = f"configs/{args.platform}/numa.json"
    models_fn = f"configs/{args.platform}/models.json"
    if args.embed:
        if args.model_size == "s":
            models_fn = f"configs/models/small_models_embed.json"
        elif args.model_size == "l":
            models_fn = f"configs/models/large_models_embed.json"
        elif args.model_size == "o":
            models_fn = f"configs/models/models_embed.json"
        else:
            models_fn = f"configs/{args.platform}/models_embed.json"
    containers_fn = f"configs/{args.platform}/containers.json"

    conf['numa'] = get_json(numa_fn)
    conf['models'] = get_json(models_fn)
    conf['containers'] = get_json(containers_fn)

    if args.benchmark:
        conf['test_type'] = 'benchmark'
    elif args.sweep:
        conf['test_type'] = 'sweep'

    if args.queries_per_concurrency:
        conf['qpc'] = args.queries_per_concurrency
    else:
        if args.embed and args.benchmark:
            conf['qpc'] = 100
        elif args.benchmark:
            conf['qpc'] = 20
        elif args.sweep:
            conf['qpc'] = 8

    if args.no_proxy:
        PROXY_ENV = ''
        conf['proxy'] = False
    else:
        conf['proxy'] = True

    if args.iterations:
        conf['iterations'] = args.iterations
    else:
        conf['iterations'] = 1

    return conf


def get_best_result(res_files, res_param, compare):
    best_res = float('inf') if compare == min else -float('inf')
    best_res_file = ""

    for f in res_files:
        r = get_json(f, fail_on_empty=False)
        if not r or 'error' in r:
            logging.warning(f"Skipping file {f} due to failed JSON parsing.")
            return r
        best_res = compare(r[res_param], best_res)
        if best_res == r[res_param]:
            best_res_file = f

    return get_json(best_res_file)


def benchmark(test, conf):
    containers_conf = conf['containers']
    qpc = conf['qpc']
    token_combinations = test['test_parameters']['benchmark_tests']
    for token_comb in token_combinations:
        res_files = run_benchmark_iters(test['model'], token_comb, containers_conf, qpc, conf['iterations'])
        results = get_best_result(res_files, 'p90_tpot_ms', min)
        token_comb['p90_op_token_throughput'] = results['output_throughput']
        token_comb['p90_ttft'] = results['p90_ttft_ms']
        token_comb['p90_tpot'] = results['p90_tpot_ms']
        token_comb['p90_itl'] = results['p90_itl_ms']
        token_comb['p90_query_lat'] = results['p90_e2el_ms']
        token_comb['p90_query_tput'] = results['request_throughput']

def benchmark_embed(test, conf, args):
    containers_conf = conf['containers']
    qpc = conf['qpc']
    token_combinations = test['test_parameters']['benchmark_tests']
    last_failure = -1
    i = 0
    while i < len(token_combinations):
        token_comb = token_combinations[i]
        res_file = run_benchmark_iters_embed(test['model'], token_comb, containers_conf, qpc, conf['iterations'])
        results = get_best_result(res_file, 'p90_e2el_ms', min)
        if 'error' in results:
            logging.warning(f"Skipping benchmark for {test['model']} due to failed JSON parsing.")
            if i < len(token_combinations) - 1:
                logging.info(f"Re-launching vllm containers for next token_comb after failure: {token_comb['inp_tokens']}, {token_comb['concurrency']}, {test['model']}")
                launch_vllm(test, conf['numa'], conf['containers'], embed=args.embed, gpu=args.gpu, trace=args.trace)
                if last_failure != i:
                    last_failure = i
                    # TODO: do we want to rerun this token_comb? Might cause dead circle
                else:
                    i += 1
            continue
        if results and 'p90_e2el_ms' in results and 'request_throughput' in results:
            token_comb['p90_query_lat'] = results['p90_e2el_ms']
            token_comb['p90_query_tput'] = results['request_throughput']
        i = i + 1


def sweep(test, conf):
    containers_conf = conf['containers']
    qpc = conf['qpc']
    token_combinations = test['test_parameters']['sweep_tests']
    for token_comb in token_combinations:
        kpi = True
        token_comb['concurrency'] = token_comb['start_concurrency']
        while kpi == True:
            res_file = run_benchmark(test['model'], token_comb, containers_conf, qpc, False)
            results = get_json(res_file)
            if results['p90_tpot_ms'] > token_comb['tpot_kpi'] or results['p90_ttft_ms'] > token_comb['ttft_kpi']:
                kpi = False
                token_comb['p90_op_token_throughput'] = results['output_throughput']
                token_comb['p90_ttft'] = results['p90_ttft_ms']
                token_comb['p90_tpot'] = results['p90_tpot_ms']
                token_comb['p90_itl'] = results['p90_itl_ms']
                token_comb['p90_query_lat'] = results['p90_e2el_ms']
                token_comb['p90_query_tput'] = results['request_throughput']
                break
            else:
                token_comb['concurrency'] += token_comb['concurrency_step']

def sweep_embed(test, conf):
    containers_conf = conf['containers']
    qpc = conf['qpc']
    token_combinations = test['test_parameters']['sweep_tests']
    for token_comb in token_combinations:
        kpi = True
        token_comb['concurrency'] = token_comb['start_concurrency']
        while kpi == True:
            res_file = run_benchmark_embed(test['model'], token_comb, containers_conf, qpc, False)
            results = get_json(res_file)
            if 'p90_e2el_ms' not in results:
                kpi = False
            elif results['p90_e2el_ms'] > token_comb['e2el_kpi']:
                kpi = False
                token_comb['p90_query_lat'] = results['p90_e2el_ms']
                token_comb['p90_query_tput'] = results['request_throughput']
                break
            else:
                token_comb['concurrency'] += token_comb['concurrency_step']

def main(args):
    #Read all configs
    if args.platform == 'g6e' and not args.gpu:
        logging.error("G6e platform requires GPU tests, use -g option to run tests on GPU")
        sys.exit(1)

    conf = get_configs(args)

    if args.model and not args.test_parameters and not args.launch_vllm:
        for m in conf['models']:
            if m['model'] == args.model:
                logging.info(m)
                sys.exit(0)

    if not args.model and args.test_parameters:
        logging.error("Specify the model for test parameters")
        sys.exit(1)

    if not args.model and args.launch_vllm:
        logging.error("Specify the model to launch vllm containers")
        sys.exit(1)

    if args.launch_vllm:
        mp = None
        for m in conf['models']:
            if m['model'] == args.model:
                mp = m
                break
        if mp == None:
            logging.error("specified model {args.model} not found in {args.platform}/models.json")
            sys.exit(1)
        launch_vllm(mp, conf['numa'], conf['containers'], embed=args.embed, gpu=args.gpu)
        logging.info("Done launching vllm containers")
        sys.exit(0)

    #Download and quantize models
    run_download(conf['containers']['dq']['image'])

    #Prepare tests, create results dir
    tests = []
    if args.model and args.test_parameters:
        tests = prepare_single_test(args.model, args.test_parameters)
    else:
        tests = prepare_tests(conf['models'])

    for test in tests:
        #Launch vllm server for first model (mount models dir and result dir for profile)
        if not args.no_launch_vllm:
            launch_vllm(test, conf['numa'], conf['containers'], embed=args.embed, gpu=args.gpu, trace=args.trace)
            if args.trace:
                logging.info("Run with vllm profiler, the benchmark result is not representative and therefore skipped")
                continue

        if args.embed and args.sweep:
            sweep_embed(test, conf)
        elif args.embed and args.benchmark:
            benchmark_embed(test, conf, args)
        elif args.benchmark:
            benchmark(test, conf)
        elif args.sweep:
            sweep(test, conf)

        if not args.no_launch_vllm:
            stop_vllm(conf['numa'])

    logging.info("--- Final test summary ---")
    if args.trace:
        logging.info("Run with vllm profiler, the benchmark result is not representative and therefore skipped")
        print(f"Traces are stored in {TRACE_DIR_BASE}")
        sys.exit(0)
    for test in tests:
        logging.info(f"  Model: {test['model']}")
        token_combinations = test['test_parameters']['benchmark_tests'] if args.benchmark else test['test_parameters']['sweep_tests']
        for token_comb in token_combinations:
            if not args.embed:
                logging.info(f"    Inp tokens: {token_comb['inp_tokens']}, Op tokens: {token_comb['op_tokens']}, Concurrency: {token_comb['concurrency']}")
                logging.info(f"      P90 token tput: {round(token_comb['p90_op_token_throughput'], 2)} tokens/sec")
                logging.info(f"      P90 Time to First Token {round(token_comb['p90_ttft'], 2)} ms")
                logging.info(f"      P90 Time per output token {round(token_comb['p90_tpot'], 2)} ms")
                logging.info(f"      P90 Inter token latency {round(token_comb['p90_itl'], 2)} ms")
            else:
                logging.info(f"    Inp tokens: {token_comb['inp_tokens']}, Concurrency: {token_comb['concurrency']}")

            if 'token_comb' in token_comb and 'p90_query_lat' in token_comb and 'p90_query_tput' in token_comb:
                logging.info(f"      P90 Query latency {round(token_comb['p90_query_lat'], 2)} ms")
                logging.info(f"      P90 Query throughput {round(token_comb['p90_query_tput'], 2)} queries/sec")
    logging.info("--- End of test summary ---")
    print(f"Results are stored in {RESULTS_DIR_BASE}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Benchmarking automation, Specify platform and test type")
    parser.add_argument("-np", "--no-proxy", help="don't pass proxy env vars to vllm container", action="store_true")
    parser.add_argument("-qpc", "--queries-per-concurrency", type=int, help="Number of queries to be sent for a given concurrency")
    parser.add_argument("-i", "--iterations", type=int, help="Number of iterations to run per test")
    parser.add_argument("-p", "--platform", choices=["spr", "gnr", "g6e", "r7i-4x", "r7i-8x", "gnr-4x", "gnr-4x-small", "spr-dev"], help="specify test platform (SPR/GNR/G6e/R7i.4x/R7i.8x)", required=True)
    parser.add_argument("-ms", "--model-size", choices=["s", "l", "o"], help="specify test model list instead of using the platform specific ones (small_size_list/large_size_list/original_list)", default="o")
    parser.add_argument("-nl", "--no-launch-vllm", help="doesn't launch or stop vllm/nginx containers. Use this to run multiple tests on prior launched vllm", action="store_true")
    parser.add_argument("-m", "--model", type=str, help="Specify model (for single model execution). If -tp is not passed, display test parameters of the model and exit")
    parser.add_argument("-tp", "--test-parameters", type=str, help="Specify test parameters in json string format for the specified model")
    parser.add_argument("-e", "--embed", help="run embedding model tests", action="store_true")
    parser.add_argument("-g", "--gpu", help="run tests on GPU", action="store_true")
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("-b", "--benchmark", help="start benchmark run", action="store_true")
    group1.add_argument("-tt", "--trace", help="run vllm profiler", action="store_true")
    group1.add_argument("-s", "--sweep", help="start sweeper run", action="store_true")
    group1.add_argument("-l", "--launch-vllm", help="only launches the vllm/nginx containers, user should stop the containers after use with docker stop", action="store_true")
    args = parser.parse_args()
    main(args)

