# Imports
import argparse
import json
import sys

# Import model
def import_model(model_cfg_file):
    with open(model_cfg_file, 'r') as f:
        model_cfg = json.load(f)
        model_type = model_cfg['type']
        if model_type == 'DLRM':
            from models.dlrm import DLRM
            model = DLRM(model_cfg)
        elif model_type == 'DLRM_Transformer':
            from models.dlrm_transformer import DLRM_Transformer
            model = DLRM_Transformer(model_cfg)
        elif model_type == 'DLRM_MoE':
            from models.dlrm_moe import DLRM_MoE
            model = DLRM_MoE(model_cfg)
        elif model_type == 'LLM':
            from models.llm import LLM
            model = LLM(model_cfg)
        elif model_type == 'LLM_MoE':
            from models.llm_moe import LLM_MoE
            model = LLM_MoE(model_cfg)
        elif model_type == 'ViT':
            from models.vit import ViT
            model = ViT(model_cfg)
        elif model_type == 'Seamless':
            from models.seamless import SeamlessM4T
            model = SeamlessM4T(model_cfg)
        else:
            sys.exit('Model type "{}" undefined!'.format(model_type))
    return model

# Import distributed system
def import_system(system_cfg_file):
    with open(system_cfg_file, 'r') as f:
        system_cfg = json.load(f)
        system_type = system_cfg['type']
        if system_type == 'GPU':
            from systems.gpus import GPUs
            system = GPUs(system_cfg)
        else:
            sys.exit('System type "{}" undefined!'.format(system_type))
    return system

# Import task
def import_task(model, system, task_cfg_file):
    with open(task_cfg_file, 'r') as f:
        task_cfg = json.load(f)
    if model.type == 'DLRM':
        from tasks.dlrm_tasks import DLRM_Task
        task = DLRM_Task(model, system, task_cfg)
    elif model.type == 'DLRM_Transformer':
        from tasks.dlrm_transformer_tasks import DLRM_Transformer_Task
        task = DLRM_Transformer_Task(model, system, task_cfg)
    elif model.type == 'DLRM_MoE':
        from tasks.dlrm_moe_tasks import DLRM_MoE_Task
        task = DLRM_MoE_Task(model, system, task_cfg)
    elif model.type == 'LLM':
        # Note that this is before optimizing FSDP prefetch communication. TODO: Adding optimized FSDP configuration.
        from tasks.llm_tasks_preoptimized import LLM_Task 
        task = LLM_Task(model, system, task_cfg)
    elif model.type == 'LLM_MoE':
        from tasks.llm_moe_tasks import LLM_MoE_Task
        task = LLM_MoE_Task(model, system, task_cfg)
    elif model.type == 'ViT':
        from tasks.vit_tasks import ViT_Task
        task = ViT_Task(model, system, task_cfg)
    elif model.type == 'Seamless':
        from tasks.seamless_tasks import SeamlessM4T_Task
        task = SeamlessM4T_Task(model, system, task_cfg)
    return task

# Parse command line configurations
def parse_configurations():
    parser = argparse.ArgumentParser(description='Performance Model')

    # Configuration Arguments
    parser.add_argument('--model-cfg-file', type=str, default='model_cfgs/dlrm/dlrm_a.json', help='Model architecture configuration file.')
    parser.add_argument('--system-cfg-file', type=str, default='system_cfgs/zionex/zionex_128.json', help='System configuration file.')
    parser.add_argument('--task-cfg-file', type=str, default='task_cfgs/dlrm/dlrm_train.json', help='Task configuration file.')
    parser.add_argument('--figures-dir', type=str, default='figures/', help='Directory to save figures.')

    return parser.parse_args()