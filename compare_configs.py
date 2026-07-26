import yaml
import json
import sys

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    if not isinstance(d, dict):
        return {parent_key: d}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    try:
        with open('config_files/configure.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading yaml: {e}")
        sys.exit(1)

    try:
        with open('config_files/wandb_config.json', 'r') as f:
            wandb_configs = json.load(f)
    except Exception as e:
        print(f"Error reading json: {e}")
        sys.exit(1)

    flat_yaml = flatten_dict(yaml_config)
    
    report = ["# Configuration Comparison: configure.yaml vs wandb_config.json\n"]
    
    for i, w_conf in enumerate(wandb_configs):
        report.append(f"## Comparison with wandb_config.json Configuration [{i+1}]\n")
        
        extracted_w_conf = {}
        for k, v in w_conf.items():
            if k == '_wandb': continue
            if isinstance(v, dict) and 'value' in v:
                extracted_w_conf[k] = v['value']
            else:
                extracted_w_conf[k] = v
                
        flat_w = flatten_dict(extracted_w_conf)
        
        all_keys = set(flat_yaml.keys()).union(set(flat_w.keys()))
        
        diffs = []
        only_in_yaml = []
        only_in_wandb = []
        
        for k in sorted(all_keys):
            if k not in flat_yaml:
                only_in_wandb.append(f"- **{k}**: (Not in configure.yaml) | wandb = `{flat_w[k]}`")
                continue
            if k not in flat_w:
                only_in_yaml.append(f"- **{k}**: yaml = `{flat_yaml[k]}` | (Not in wandb_config.json)")
                continue
                
            v_yaml = flat_yaml[k]
            v_w = flat_w[k]
            
            # Simple conversion for comparison
            str_v_yaml = str(v_yaml).lower().replace(' ', '')
            str_v_w = str(v_w).lower().replace(' ', '')
            
            # Handle float vs int, list vs string differences
            if str_v_yaml != str_v_w:
                # Let's try to parse both as floats and check if they are equal
                is_num_equal = False
                try:
                    if float(v_yaml) == float(v_w):
                        is_num_equal = True
                except:
                    pass
                if not is_num_equal:
                    diffs.append(f"- **{k}**: `configure.yaml` = `{v_yaml}` | `wandb_config.json` = `{v_w}`")
                
        if diffs:
            report.append("### Value Differences:\n")
            report.extend(diffs)
            report.append("\n")
            
        if only_in_yaml:
            report.append("### Only in configure.yaml:\n")
            report.extend(only_in_yaml)
            report.append("\n")
            
        if only_in_wandb:
            report.append("### Only in wandb_config.json:\n")
            report.extend(only_in_wandb)
            report.append("\n")
            
        if not diffs and not only_in_yaml and not only_in_wandb:
            report.append("No differences found.\n\n")
            
    with open('/home/ai2lab/.gemini/antigravity-ide/brain/464a02d9-0725-44c7-8ab2-874faa71d747/comparison_results.md', 'w') as f:
        f.write("\n".join(report))
        
    print("Comparison complete. Wrote to comparison_results.md")

if __name__ == '__main__':
    main()
