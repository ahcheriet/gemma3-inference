import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

from unsloth import FastLanguageModel
import torch
import re
import json

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "gemma3_kimi_lora",
    max_seq_length = 2048,
    load_in_4bit = True,
    device_map = "cuda:0",
)
FastLanguageModel.for_inference(model)

prompt = '''<bos><start_of_turn>user
System: You are a workflow engine controller that prepares task inputs. Always respond with valid JSON only.

You are a workflow engine controller. Prepare the input for the next task.
WORKFLOW: onboarding
Description: Processus d onboarding bancaire : vérification du dossier client, contrôle responsable, scoring d interdiction automatique (DMN), décision manuelle éventuelle, appel aux APIs core banking, et vérification manuelle d intégration. En cas de réponse défavorable, le dossier peut être résoumis ou rejeté définitivement.
Task: HUMAN (ID: Controle Juriste)
Description: Contrôle du juriste sur la conformité légale du dossier
Expected Input Schema:
{"type": "object", "required": ["customer", "account", "employee_approval"], "properties": {"account": {"type": "object"}, "form_id": {"type": "string"}, "customer": {"type": "object"}, "department": {"type": "object"}, "employee_approval": {"type": "string"}}}
Global Context:
{"_task_comments": ["test"],"account": {"account_number": "0040023232120329293","acctype": {"label": "Compte Epargne","ledger": "220900"},"acount_accuracy": "string","acount_agence": "1001","acount_approbation_date": "string","acount_approbation_par": "string","acount_cif": "string","acount_compte_joint": "string","acount_cree_par": "string","acount_datecreation": "string","acount_departement": "string","acount_devision": "string","acount_gl_code": "220110","acount_secteur_economique_sec": "0111 - Culture de céréales (blé, orge…)","acount_secteur_econpmique": "01 : Cultures agricoles et production végétale","acount_solde_min_releve": "string","acount_type_compte": "Compte Courrant"},"accuracy": 0,"approved": true,"assignee": "$assignee","auto_interdiction": false,"cif": {"agence": "1001","num_cif": "109203","type_cif": "Persone Physique"},"core_logs": {"coreLogs": "no log","coreProblem": false},"customer": {"address": {"country": "string","district": "string","state": "string","street": "string","zipCode": "string"},"birthDate": "1982-04-21T00:00:00.000","birthPlace": {"country": "Algeria","district": "string","state": "string","street": "string","zipCode": "string"},"customerType": {"personCategory": {"label": "string"},"personType": {"label": "string"},"profession": {"profession": "string"}},"documents": [{"document_category": "string","document_type": "string"}],"firstName_ar": "احمد","firstName_fr": "Ahmed","interdictions": [{"flag": true,"interdiction_type": "string"}],"lastName_ar": "حكيم","lastName_fr": "Hakim","nationality": {"label": "string"},"nin": "121123123123123121212","profession": {"profession": "string"},"prohibited": true},"department": {"codes": ["1001"]},"employee_approval": false,"form_id": "ab3e3ab8-38e7-46a6-9900-66736a722946","input_error": true,"lang": "en","manager_approval": false,"manual_interdiction": "$manual_interdiction","owner_groups": ["CustRelations"],"owner_id": "custrel_1001","reviewer_notes": "Yes","status": "initialized","task_id": "start"}
Previous Task Output:
{"_task_comments": ["test"],"approved": true,"reviewer_notes": "Yes"}
EXECUTION HISTORY:
[{"chronological_order": 3,"completed_at": "2026-04-24T20:35:29Z","task_id": "Verification Dossier","task_output": {"approved": true,"reviewer_notes": "Yes"}},{"chronological_order": 4,"completed_at": "2026-04-24T20:41:44Z","task_id": "Verification Dossier","task_output": {"approved": true,"reviewer_notes": "Yes"}}]
Generate a JSON object that conforms to the Expected Input Schema. Return ONLY a valid JSON object with no markdown formatting.<end_of_turn>
<start_of_turn>model
'''

inputs = tokenizer([prompt], return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 1024,
        temperature = 0.1,
        do_sample = True,
        pad_token_id = tokenizer.eos_token_id,
    )

generated = outputs[0][inputs['input_ids'].shape[1]:]
response = tokenizer.decode(generated, skip_special_tokens=True)

# Strip <think> block
response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

# Validate JSON
try:
    parsed = json.loads(response)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
except json.JSONDecodeError as e:
    print(f"Warning: output is not valid JSON: {e}")
    print(response)