"""Code Understanding - Tree-sitter + regex fallback for 9 languages."""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "java", "go", "rust", "c", "cpp", "ruby"]


@dataclass
class CodeBlock:
    """Extracted code block with metadata."""
    name: str = ""
    kind: str = ""  # function/class/method/module/variable
    language: str = ""
    source: str = ""
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""
    params: list[str] = field(default_factory=list)
    returns: str = ""
    complexity: int = 1  # cyclomatic complexity


class CodeUnderstandingEngine:
    """Parse code using tree-sitter (if available) or regex fallback."""
    
    def __init__(self) -> None:
        self._ts_available = False
        self._parsers: dict[str, Any] = {}
        try:
            import tree_sitter_languages
            self._ts_available = True
            self._tsl = tree_sitter_languages
        except ImportError:
            logger.info("tree-sitter-languages not available, using regex fallback")
    
    def parse(self, code: str, language: str = "python") -> list[CodeBlock]:
        """Parse code into structured blocks."""
        language = language.lower()
        if language not in SUPPORTED_LANGUAGES:
            language = "python"
        
        if self._ts_available:
            try:
                return self._parse_treesitter(code, language)
            except Exception as e:
                logger.warning(f"Tree-sitter parse error, falling back: {e}")
        
        return self._parse_regex(code, language)
    
    def _parse_treesitter(self, code: str, language: str) -> list[CodeBlock]:
        """Parse using tree-sitter for accurate AST extraction."""
        parser = self._get_parser(language)
        tree = parser.parse(code.encode())
        blocks = []
        self._walk_tree(tree.root_node, code, language, blocks)
        return blocks
    
    def _get_parser(self, language: str) -> Any:
        if language not in self._parsers:
            lang = self._tsl.get_language(language)
            parser = self._tsl.get_parser(language)
            self._parsers[language] = parser
        return self._parsers[language]
    
    def _walk_tree(self, node: Any, code: str, language: str, blocks: list[CodeBlock]) -> None:
        """Walk tree-sitter AST to extract code blocks."""
        kind_types = {
            "python": {"function_definition", "class_definition"},
            "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
            "java": {"method_declaration", "class_declaration", "interface_declaration"},
            "go": {"function_declaration", "method_declaration"},
            "rust": {"function_item", "impl_item", "struct_item"},
            "c": {"function_definition", "struct_specifier"},
            "cpp": {"function_definition", "class_specifier", "struct_specifier"},
            "ruby": {"method", "class", "module"},
        }
        
        target_types = kind_types.get(language, {"function_definition", "class_definition"})
        
        if node.type in target_types:
            name = self._extract_name(node, code)
            source = code[node.start_byte:node.end_byte]
            docstring = self._extract_docstring(node, code, language)
            params = self._extract_params(node, code)
            complexity = self._estimate_complexity(source)
            
            blocks.append(CodeBlock(
                name=name, kind=node.type.replace("_definition", "").replace("_declaration", "").replace("_item", ""),
                language=language, source=source,
                start_line=node.start_point[0], end_line=node.end_point[0],
                docstring=docstring, params=params, complexity=complexity,
            ))
        
        for child in node.children:
            self._walk_tree(child, code, language, blocks)
    
    def _extract_name(self, node: Any, code: str) -> str:
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
                return code[child.start_byte:child.end_byte]
        return ""
    
    def _extract_docstring(self, node: Any, code: str, language: str) -> str:
        body = None
        for child in node.children:
            if child.type in ("block", "body", "statement_block"):
                body = child
                break
        if body and body.children:
            first = body.children[0]
            if first.type in ("string", "string_literal", "expression_statement"):
                text = code[first.start_byte:first.end_byte]
                return text.strip("\"'` \n")
        return ""
    
    def _extract_params(self, node: Any, code: str) -> list[str]:
        for child in node.children:
            if child.type in ("parameters", "parameter_list", "argument_list"):
                return [code[p.start_byte:p.end_byte] for p in child.children if p.type in ("identifier", "parameter", "typed_parameter")]
        return []
    
    def _estimate_complexity(self, source: str) -> int:
        """Estimate cyclomatic complexity."""
        keywords = ["if ", "elif ", "else:", "for ", "while ", "except ", "and ", "or ", "case "]
        return 1 + sum(source.count(kw) for kw in keywords)
    
    def _parse_regex(self, code: str, language: str) -> list[CodeBlock]:
        """Regex-based fallback parser."""
        blocks = []
        if language == "python":
            # Match functions
            for m in re.finditer(r'^(\s*)(def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:)', code, re.MULTILINE):
                indent = len(m.group(1))
                name = m.group(3)
                params = [p.strip() for p in m.group(4).split(",") if p.strip()]
                start = code[:m.start()].count('\n')
                # Find end
                end = start
                lines = code.split('\n')
                for i in range(start + 1, len(lines)):
                    if lines[i].strip() and not lines[i].startswith(' ' * (indent + 1)) and not lines[i].startswith('\t' * (indent // 4 + 1)):
                        break
                    end = i
                source = '\n'.join(lines[start:end + 1])
                blocks.append(CodeBlock(
                    name=name, kind="function", language=language, source=source,
                    start_line=start, end_line=end, params=params,
                    complexity=self._estimate_complexity(source),
                ))
            # Match classes
            for m in re.finditer(r'^class\s+(\w+)', code, re.MULTILINE):
                start = code[:m.start()].count('\n')
                name = m.group(1)
                blocks.append(CodeBlock(
                    name=name, kind="class", language=language, source=m.group(0),
                    start_line=start, end_line=start,
                    complexity=1,
                ))
        elif language in ("javascript", "typescript"):
            for m in re.finditer(r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()', code):
                name = m.group(1) or m.group(2)
                start = code[:m.start()].count('\n')
                blocks.append(CodeBlock(name=name, kind="function", language=language, source=m.group(0), start_line=start, end_line=start, complexity=1))
        elif language == "go":
            for m in re.finditer(r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(', code):
                name = m.group(1)
                start = code[:m.start()].count('\n')
                blocks.append(CodeBlock(name=name, kind="function", language=language, source=m.group(0), start_line=start, end_line=start, complexity=1))
        elif language == "rust":
            for m in re.finditer(r'(?:pub\s+)?fn\s+(\w+)', code):
                name = m.group(1)
                start = code[:m.start()].count('\n')
                blocks.append(CodeBlock(name=name, kind="function", language=language, source=m.group(0), start_line=start, end_line=start, complexity=1))
        else:
            # Generic: match any word followed by parens
            for m in re.finditer(r'(\w+)\s*\([^)]*\)\s*\{', code):
                name = m.group(1)
                if name not in ("if", "for", "while", "switch", "catch"):
                    start = code[:m.start()].count('\n')
                    blocks.append(CodeBlock(name=name, kind="function", language=language, source=m.group(0), start_line=start, end_line=start, complexity=1))
        return blocks
    
    def detect_language(self, code: str, filename: str = "") -> str:
        """Detect programming language from code content or filename."""
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".go": "go", ".rs": "rust",
            ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
            ".rb": "ruby",
        }
        if filename:
            for ext, lang in ext_map.items():
                if filename.endswith(ext):
                    return lang
        
        # Heuristic detection
        if "def " in code and "import " in code:
            return "python"
        if "function " in code and ("var " in code or "const " in code):
            return "javascript"
        if "func " in code and ":=" in code:
            return "go"
        if "fn " in code and "let " in code and "mut " in code:
            return "rust"
        if "public class" in code or "private void" in code:
            return "java"
        return "python"
