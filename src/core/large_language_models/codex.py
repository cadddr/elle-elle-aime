from dotenv import dotenv_values
import openai
import whatthepatch
import pprint
import lang_utils

config = dotenv_values(".env")
openai.api_key = config.get('OPENAI_API_KEY')

EXAMPLE_FILE_PATH = "/Users/pengyu/src/kth/plm-repair-them-all/data/example/Fibonacci.java"
EXAMPLE_BUGGY_FILE_PATH = "/Users/pengyu/src/kth/repair/Defects4J_Closure_3/src/com/google/javascript/jscomp/FlowSensitiveInlineVariables.java"
EXAMPLE_PATCH_FILE_PATH = "/Users/pengyu/src/kth/plm-repair-them-all/benchmarks/defects4j/framework/projects/Closure/patches/3.src.patch"
MAX_TOKEN_LENGTH = 3570
CODE_TOO_LONG = "Code is too long"
CODEX_MODEL = "code-davinci-002"


def add_prompt_to_code(code):
    return "##### Fix bugs in the below function\n \n### Buggy Java\n" + code + "\n### Fixed Java\n"


def request_codex_code_complition(code):
    # https://beta.openai.com/docs/api-reference/completions/create
    response = openai.Completion.create(
        model=CODEX_MODEL,
        prompt=code,
        temperature=0,
        max_tokens=MAX_TOKEN_LENGTH,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"]
    )
    return response.choices[0].text  # type: ignore


def repair_code(code):
    # add prompt
    code = add_prompt_to_code(code)
    token_length = len(code.split())

    if token_length > MAX_TOKEN_LENGTH:
        print(CODE_TOO_LONG)
        return
    else:
        print('token length: ', token_length)
        response = request_codex_code_complition(code)
        return response


def load_patch_file(file_path):
    countable_changes = []
    with open(file_path, 'r') as file:
        text = file.read()
    for diff in whatthepatch.parse_patch(text):
        pp = pprint.PrettyPrinter(indent=4)
        for change in diff.changes or []:
            if change.new == None and lang_utils.is_line_countable(change.line):
                countable_changes.append(change.old)
        pp.pprint(countable_changes)

    return countable_changes


def main():
    changes = load_patch_file(EXAMPLE_PATCH_FILE_PATH)
    the_method = lang_utils.load_patch_code_snippets(
        EXAMPLE_BUGGY_FILE_PATH, changes)
    print('--------------------------buggy code--------------------------')
    print(the_method)
    code = lang_utils.clean_code(the_method.code_snippet)
    fixed_code = repair_code(code)
    print('--------------------------fixed code--------------------------')
    print(fixed_code)


if __name__ == "__main__":
    main()