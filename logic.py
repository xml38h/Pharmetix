def run_model_on_input(input_list):
    results = []

    for txt in input_list:
        output = {
            "input_text": txt,
            "drug_likeness": "Yes",
            "details": "Example output (replace with real model output)"
        }
        results.append(output)

    return results
