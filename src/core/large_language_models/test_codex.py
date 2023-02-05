
from core.large_language_models.codex import load_code_node, sanitize_choice_text
from core.tools.patch import read_patch_file


def test_load_buggy_code_node():
    fixed_file_path = "src/fixtures/Defects4J_Closure_01_fixed.source"
    buggy_file_path = "src/fixtures/Defects4J_Closure_01_buggy.source"
    patch_file_path = 'src/fixtures/Defects4J_Closure_01.patch'
    countable_diffs, patch_text = read_patch_file(patch_file_path)
    fixed_node, buggy_node = load_code_node(
        fixed_file_path, buggy_file_path, countable_diffs)
    assert buggy_node.name == 'removeUnreferencedFunctionArgs'
    assert buggy_node.type == 'MethodDeclaration'
    assert buggy_node.start_pos == 369
    assert buggy_node.end_pos == 406
    print('buggy code body:\n', buggy_node.code_lines_str())


def test_sanitize_choice_text():
    dirty_file_path = "src/fixtures/Defects4J_Chart_01_codex_response_dirty.txt"
    with open(dirty_file_path, 'r') as file:
        dirty_text = file.read()
    clean_file_path = "src/fixtures/Defects4J_Chart_01_codex_response_clean.txt"
    with open(clean_file_path, 'r') as file:
        clean_text = file.read()
    assert clean_text == sanitize_choice_text(dirty_text)