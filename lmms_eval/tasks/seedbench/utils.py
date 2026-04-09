import json

# SEED-Bench-1 question_type_id → evaluation dimension mapping
# Source: https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md
QUESTION_TYPE_MAP = {
    1: "scene_understanding",
    2: "instance_identity",
    3: "instance_attributes",
    4: "instance_location",
    5: "instances_counting",
    6: "spatial_relation",
    7: "instance_interaction",
    8: "visual_reasoning",
    9: "text_understanding",
    10: "action_recognition",
    11: "action_prediction",
    12: "procedure_understanding",
}


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return (
        f"{question}\nAnswer with the option's letter from the given choices directly."
    )


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"]
    result_data = {
        "pred": pred,
        "answer": answer,
        "question_id": doc["question_id"],
    }

    results = {
        f"seed_{data_type}": result_data,
        "seed_all": result_data,
    }

    # Add per-question-type metric
    question_type_id = doc.get("question_type_id")
    if question_type_id is not None and question_type_id in QUESTION_TYPE_MAP:
        dimension = QUESTION_TYPE_MAP[question_type_id]
        results[f"seed_{dimension}"] = result_data

    return results


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append(
            {"question_id": result["question_id"], "prediction": result["pred"]}
        )
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score


def seed_doc_to_text_mc(doc):
    question = doc["question"]
    return f"{question} Answer :"


def seed_doc_to_choice(doc):
    return [doc["choice_a"], doc["choice_b"], doc["choice_c"], doc["choice_d"]]


def seed_doc_to_mc_target(doc):
    answer2choice = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
    return doc[answer2choice[doc["answer"]]]
