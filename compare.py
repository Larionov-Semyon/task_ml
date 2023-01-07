import argparse
import ast
from io import StringIO
# import tokenize

parser = argparse.ArgumentParser(description='Antiplagiat')
parser.add_argument('input', type=str, help='input files')
parser.add_argument('score', type=str, help='score files')
args = parser.parse_args()


class TransformerRemoveDoc(ast.NodeTransformer):
    """Class for remove docstring"""
    def visit_Module(self, node):
        self.generic_visit(node)
        return self._visit_docstring_parent(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._visit_docstring_parent(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._visit_docstring_parent(node)

    def _visit_docstring_parent(self, node):
        """Remove first node if it's docstring"""
        new_body = []
        for i, child_node in enumerate(node.body):
            if i == 0 and isinstance(child_node, ast.Expr) and isinstance(child_node.value, ast.Str):
                pass
            else:
                new_body.append(child_node)
        node.body = new_body
        return node


class TransformerToken(ast.NodeTransformer):
    """Class for code tokenization"""

    def __init__(self):
        self._arg_count = 0

    def visit_ClassDef(self, node):
        """Change class name"""
        node.name = 'class_name'
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        """Change function name, arguments and remove annotations"""
        node.name = "function_name "
        node.returns = None
        if node.args.args:
            for arg in node.args.args:
                arg.annotation = None
                arg.arg = 'param'
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        """Remove import"""
        return None

    def visit_ImportFrom(self, node):
        """Remove import"""
        return None

    def visit_Call(self, node):
        """Remove keywords"""
        for i in node.keywords:
            i.arg = 'param'
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        """Change async function name"""
        node.name = "asyncfunction_name "
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        """Change arg name"""
        node.name = "arg"
        self.generic_visit(node)
        return node

    def visit_arguments(self, node):
        """Change arg name"""
        node.name = "arg"
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        """Change attribute name"""
        node.name = "arg"
        self.generic_visit(node)
        return node

    def visit_Num(self, node):
        """Replace num by 1"""
        node.n = 1
        return node

    def visit_Str(self, node):
        """Replace str by 'string' """
        node.n = 'string'
        return node

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id='param'), node)


def distance(lst1, lst2):
    """Levenshtein distance (standard realization)"""
    n, m = len(lst1), len(lst2)
    if n < m:
        a, b = lst1, lst2
    else:
        a, b = lst2, lst1
        n, m = m, n

    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
    return current_row[n]


def check_plagiat(code1, code2):
    score = distance(code1, code2) / max(len(code1), len(code2))
    return score


def normalize(text):
    """Normalization. (comments aren't included in ast)"""
    text = text.lower().rstrip()
    tree = ast.parse(text)
    optimizer = TransformerRemoveDoc()
    tree = optimizer.visit(tree)
    return ast.unparse(tree)

# def byte_tokenize(file):
#     tokens = tokenize.generate_tokens(StringIO(file).readline)
#     ans = [int(token[0]) for token in tokens]
#     return ans


def change_tokenize(tree):
    """Tokenization"""
    optimizer = TransformerToken()
    tree = optimizer.visit(tree)
    return tree


def transform(path_doc):
    """
    Normalization ant tokenization.
    Remove comments, docstrings, pass, \n, gaps and etc.
    Change all names to the same format
    """
    with open(path_doc, 'r', encoding="utf8") as f:
        file = f.read()

    file = normalize(file)
    # token_byte = byte_tokenize(path_doc)
    my_tree = ast.parse(file)
    my_tree = change_tokenize(my_tree)
    # print(ast.dump(my_tree, indent=4))
    new_file = ast.unparse(my_tree)

    new_file = StringIO(new_file.replace('pass', '').replace('\n\n', '\n')).readlines()
    # new_file = StringIO(new_file.replace('\n\n', '\n')).readlines()
    new_file = map(lambda x: x.lstrip().rstrip(), new_file)
    new_file = [i for i in new_file if i != '']
    return new_file


def read_input(path):
    with open(path, 'r', encoding="utf8") as f:
        return map(lambda x: x.rstrip().split(), f.readlines())


def write_score(path, scores):
    with open(path, 'w', encoding='utf8') as f:
        f.writelines('\n'.join(map(lambda x: str(round(x, 3)), scores)))


def main():
    files = read_input(args.input)
    scores = []
    for doc1, doc2 in files:
        code1 = transform(doc1)
        code2 = transform(doc2)
        scores.append(check_plagiat(code1, code2))
    # print(scores)
    write_score(args.score, scores)


if __name__ == '__main__':
    main()
