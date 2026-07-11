"""Build provenance-complete independent-generation requests and ingest actual responses.

No provider is called by default. `build-requests` creates a JSONL job manifest that
can be submitted to any model adapter; `ingest` verifies returned model identity and
writes the validation-corpus schema. This separation prevents placeholder text from
becoming a scientific dataset.
"""
from __future__ import annotations
import argparse, csv, json
from itertools import product
from pathlib import Path
from validation_framework import REQUIRED_COLUMNS, ProvenanceError, validate_records

def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_jsonl(path, rows): Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
def read_jsonl(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def build_requests(config):
    if config.get("status") == "executed": raise ProvenanceError("Use a new configuration for a new execution; do not overwrite executed provenance.")
    needed=("generators","prompt_families","topics","temperatures","diversity_settings","talking_point_repertoires","style_instructions","automated_transformations","seed")
    missing=[key for key in needed if not config.get(key)]
    if missing: raise ProvenanceError(f"Generation configuration is missing: {', '.join(missing)}")
    requests=[]
    for gen,prompt,topic,temp,diversity,repertoire,style,transform in product(config["generators"],config["prompt_families"],config["topics"],config["temperatures"],config["diversity_settings"],config["talking_point_repertoires"],config["style_instructions"],config["automated_transformations"]):
        if gen.get("model","").startswith("REPLACE_") or gen.get("revision","").startswith("REPLACE_"):
            raise ProvenanceError("Replace every generator model ID and immutable revision before building requests.")
        request_id=f"req_{len(requests):05d}"
        instruction=(f"Write one social-media advocacy post about {topic}. Campaign objective: {prompt['objective']}. "
                     f"Use this independent talking-point repertoire: {repertoire['name']} ({'; '.join(repertoire['points'])}). "
                     f"Style instruction: {style['instruction']}. Do not copy examples or templates from any other source.")
        requests.append({"request_id":request_id,"prompt":instruction,"generator_id":gen["generator_id"],"generator_family":gen["generator_family"],"generator_model":gen["model"],"generator_revision":gen["revision"],"prompt_family":prompt["id"],"prompt_revision":prompt["revision"],"topic":topic,"temperature":str(temp),"diversity_setting":str(diversity),"talking_point_repertoire":repertoire["name"],"style_instruction":style["id"],"transformation_id":transform["id"],"transformation_revision":transform["revision"],"seed":str(config["seed"]),"status":"pending_external_execution"})
    return requests

def ingest_responses(requests,responses):
    by_id={request["request_id"]:request for request in requests}
    if len(by_id)!=len(requests): raise ProvenanceError("Request identifiers must be unique.")
    records=[]
    for response in responses:
        request=by_id.get(response.get("request_id"))
        if request is None: raise ProvenanceError(f"Response has unknown request_id {response.get('request_id')!r}.")
        if not response.get("text","").strip(): raise ProvenanceError(f"Response {request['request_id']} has no generated text.")
        if response.get("generator_model")!=request["generator_model"] or response.get("generator_revision")!=request["generator_revision"]:
            raise ProvenanceError("Response model ID/revision does not match the request.")
        records.append({"record_id":response.get("record_id",request["request_id"]),"account_id":response["account_id"],"campaign_id":response["campaign_id"],"class_label":response["class_label"],"text":response["text"],"topic":request["topic"],"generator_id":request["generator_id"],"generator_family":request["generator_family"],"generator_model":request["generator_model"],"generator_revision":request["generator_revision"],"prompt_family":request["prompt_family"],"prompt_revision":request["prompt_revision"],"temperature":request["temperature"],"diversity_setting":request["diversity_setting"],"talking_point_repertoire":request["talking_point_repertoire"],"style_instruction":request["style_instruction"],"transformation_id":request["transformation_id"],"transformation_revision":request["transformation_revision"],"evaluation_batch":response["evaluation_batch"],"split_role":response.get("split_role","evaluation"),"seed":request["seed"]})
    validate_records(records)
    return records

def main():
    parser=argparse.ArgumentParser(description="Prepare or ingest independent CCSF generation jobs.")
    sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build-requests"); build.add_argument("config"); build.add_argument("--output",required=True)
    ingest=sub.add_parser("ingest"); ingest.add_argument("requests"); ingest.add_argument("responses"); ingest.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="build-requests":
        requests=build_requests(load_json(args.config)); write_jsonl(args.output,requests); print(f"Wrote {len(requests)} pending external-generation requests to {args.output}.")
    else:
        records=ingest_responses(read_jsonl(args.requests),read_jsonl(args.responses))
        with Path(args.output).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=REQUIRED_COLUMNS); writer.writeheader(); writer.writerows(records)
        print(f"Wrote {len(records)} provenance-validated corpus records to {args.output}.")
if __name__=="__main__": main()
