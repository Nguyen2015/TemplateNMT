import copy
import logging
import re
from typing import List

logger = logging.getLogger()


class DifferenceCode:
    PREDICATE = 0
    VARIABLE = 1
    CONSTANT = 2
    CHILD_COUNT = 3


class LogicElement:
    DEFAULT_RELAX_CHILD_ORDER = {"and", "or", "", "next_to"}
    DEFAULT_ALLOW_CHILD_DUPLICATION = {"and", "or", ""}

    def __init__(self, value="", child=None, relax_child_order=False, allow_child_duplication=False, depth_level=0):
        self.child = child or []
        self.value = str(value)
        if value in LogicElement.DEFAULT_RELAX_CHILD_ORDER:
            relax_child_order = True
        self.relax_child_order = relax_child_order

        if value in LogicElement.DEFAULT_ALLOW_CHILD_DUPLICATION:
            allow_child_duplication = True
        self.allow_child_duplication = allow_child_duplication

        self.leaf_nodes = None
        self.depth_level = depth_level
        self.options = {}

    def set_option(self, k, v):
        self.options[k] = v

    def get_option(self, k):
        return self.options.get(k)

    def add_child(self, child):
        if isinstance(child, LogicElement):
            self.child.append(child)
        else:
            logger.warning("Can't add child that is not object LogicElement: {}" % {child})

    def is_variable_node(self, term_check=r'[\$\?][^\s\n]*'):
        if len(self.child) == 0 and re.fullmatch(term_check, self.value):
            return True
        else:
            return False

    def is_constant(self, term_check_variable=r'[\$\?][^\s\n]*'):
        if len(self.child) == 0 and not self.is_variable_node(term_check_variable):
            return True
        else:
            return False

    def is_triple(self):
        if len(self.child) > 0 and len(self.value) > 0:
            return True
        else:
            return False

    def is_leaf_node(self):
        # if node that have nephew node
        for ee in self.child:
            if isinstance(ee, LogicElement) and len(ee.child) > 0:
                return False
        return True

    def get_leaf_nodes(self):
        tmp_leaf_nodes = []
        if self.leaf_nodes is not None:
            return self.leaf_nodes
        elif self.is_leaf_node():
            tmp_leaf_nodes = [self]
        elif self.leaf_nodes is None:
            for e in self.child:
                tmp_leaf_nodes += e.get_leaf_nodes()
        self.leaf_nodes = tmp_leaf_nodes
        return tmp_leaf_nodes

    def get_all_node_name(self):
        node_names = []
        if len(self.value) > 0 and len(self.child) > 0:
            node_names.append(self.value)
        for e in self.child:
            node_names = node_names + e.get_all_node_name()
        return node_names

    def get_leaf_nodes_with_depth(self, depth=5):
        tmp_leaf_nodes = []
        if len(self.child) == 0 or self.depth_level >= depth:
            tmp_leaf_nodes = [self]
        else:
            for e in self.child:
                tmp_leaf_nodes += e.get_leaf_nodes_with_depth(depth=depth)
        return tmp_leaf_nodes

    def get_leaf_nodes_template(self):
        tmp_leaf_nodes = []
        if len(self.child) == 0:
            tmp_leaf_nodes = [self]
        else:
            for e in self.child:
                tmp_leaf_nodes += e.get_leaf_nodes_template()
        return tmp_leaf_nodes

    def get_max_depth(self):
        max_depth = self.depth_level
        if len(self.child) == 0:
            return max_depth
        else:
            for e in self.child:
                max_depth = max(e.get_max_depth(), max_depth)
        return max_depth

    def get_min_depth(self):
        # print(len(self.child))
        if len(self.child) == 0:
            return self.depth_level
        else:
            min_depth = 10000
            for e in self.child:
                # if re.search(r'[a-zA-Z]', e.value) is not None:
                min_depth = min(e.get_min_depth(), min_depth)
            return min_depth

    def get_triple_name(self):
        tmp_triple_name = []
        if self.is_triple():
            tmp_triple_name.append(self.value)

        if len(self.child) > 0:
            for e in self.child:
                tmp_triple_name += e.get_triple_name()

        return tmp_triple_name

    def get_constant(self):
        tmp_triple_name = []
        if self.is_constant():
            return [self.value]
        for e in self.child:
            tmp_triple_name += e.get_constant()

        return tmp_triple_name

    def prune_tag(self, tags=[]):
        if self.value in tags:
            self.child = []
        else:
            for e in self.child:
                e.prune_tag(tags=tags)
        return

    def prune_tag_novp_in_subtree(self, tags=[]):
        checking_subtree = False
        for e in self.child:
            if e.get_option('flag_vp_in_subtree') == True:
                checking_subtree = True
                break 

        if self.value == "VP" and self.value in tags and not checking_subtree:
            self.child = []
        elif self.value in tags and self.get_option('flag_vp_in_subtree') != True:
            self.child = []
        else:
            for e in self.child:
                e.prune_tag_novp_in_subtree(tags=tags)

    def flag_vp_in_subtree(self):
        if self.value.startswith("V"):
            self.set_option('flag_vp_in_subtree', True)
        for e in self.child:
            flag = e.flag_vp_in_subtree()
            if flag is True:
                self.set_option('flag_vp_in_subtree', True)
        return self.get_option('flag_vp_in_subtree')

    def flag_frequent_postag(self, common_postags):
        if self.value in common_postags:
            self.set_option('common_postags', True)
        for e in self.child:
            e.flag_frequent_postag(common_postags)

    def scan_frequent_tree(self):
        nodes_return = []

        # if one frequent postag node contains more than 2 child or word node
        b_check_pos_node = (self.get_option('common_postags') and len(self.child) > 1)
        if b_check_pos_node or len(self.child) == 0:
            nodes_return.append(self)

        for e in self.child:
            nodes_return = nodes_return + e.scan_frequent_tree()

        # if b_check_pos_node:
        #     nodes_return.append(LogicElement("/{}".format(self.value)))

        return nodes_return

    @staticmethod
    def _collapse_list_logic(logics: List) -> List:
        len_logic = len(logics)
        mask_remove = [False] * len_logic
        for i in range(len_logic - 1):
            for j in range(i + 1, len_logic):
                if not mask_remove[j]:
                    if logics[i] == logics[j]:
                        mask_remove[j] = True
        for j in range(len_logic - 1, -1, -1):
            if mask_remove[j]:
                logics.pop(j)

        return logics

    def __eq__(self, other):
        if not isinstance(other, LogicElement) or not self.value == other.value:
            return False

        if self.allow_child_duplication and self.relax_child_order:
            self_child = copy.deepcopy(self.child)
            other_child = copy.deepcopy(other.child)

            self_child = self._collapse_list_logic(self_child)
            other_child = self._collapse_list_logic(other_child)
        else:
            self_child = self.child
            other_child = other.child

        if not len(self_child) == len(other_child):
            return False
        else:
            if not self.relax_child_order:
                for i, e in enumerate(other_child):
                    if not e == self_child[i]:
                        return False
            else:
                for i, e in enumerate(other_child):
                    j = 0
                    for _, e2 in enumerate(self_child):
                        if e == e2:
                            break
                        j += 1
                    if j == len(self_child):
                        return False
            return True

    def get_path_to_leaf_nodes(self, path_to_leaf_nodes=None, cur_path=None):
        path_to_leaf_nodes = [] if path_to_leaf_nodes is None else path_to_leaf_nodes
        cur_path = [] if cur_path is None else copy.deepcopy(cur_path)
        if len(self.value) > 0:
            cur_path.append(self.value)
        if self.is_leaf_node():
            path_to_leaf_nodes.append(cur_path)
        else:
            for e in self.child:
                e.get_path_to_leaf_nodes(path_to_leaf_nodes, cur_path)
        return path_to_leaf_nodes

    def __str__(self):
        if len(self.child) == 0:
            return self.value
        child_str = " ".join([str(v) for v in self.child])
        if len(self.value) > 0 and len(child_str) > 0:
            return "( {} {} )".format(self.value, child_str)
        elif len(self.value) > 0:
            return "( {} )".format(self.value)
        elif len(child_str) > 0:
            return "( {} )".format(child_str)
        else:
            return "( )"

    @staticmethod
    def _norm_variable_name(name):
        if "$" in name:
            name = name.replace("$", "s")
        if "?" in name:
            name = name.replace("?", "") + "1"
        return name

    @staticmethod
    def _norm_predicate(name):
        if name == "":
            name = 'rootNode'
        name = name.replace(">", "Greater")
        name = name.replace("<", "Less")
        name = name.replace("<", "Less")
        return re.sub(r'[\.:_]', "-", name)

    @staticmethod
    def _norm_constant(value):
        value = re.sub(r'[\"]', "-", value)
        # value = re.sub(r'[^a-zA-Z\d]', "-", value)
        if re.search(u'[\u4e00-\u9fff]', value):
            value = "chinese-" + re.sub(u'[^a-zA-Z\d]', '-', value)
        value = "\"{}\"".format(value)
        return value

    def to_amr(self, var_exist=None):
        var_exist = var_exist or {}

        if self.is_constant():
            return self._norm_constant(self.value)
        elif self.is_variable_node():
            if var_exist is not None and self.value not in var_exist:
                var_exist[self.value] = self._norm_variable_name(self.value)
                return "({} / var)".format(self._norm_variable_name(self.value))
            else:
                return self._norm_variable_name(self.value)
        else:
            if self.value == "" and len(self.child) == 1:
                amr_str = self.child[0].to_amr(var_exist)
                if amr_str == "\"\"" or amr_str == "":
                    return "(n0 / errorRootNode)"
                elif re.fullmatch(r'".+"', amr_str):
                    return "(n0 / {})".format(amr_str)
                else:
                    return amr_str

            node_name = 'n' + str(len(var_exist))
            amr_str = "({} / {} ".format(node_name, self._norm_predicate(self.value))
            var_exist[node_name] = self._norm_predicate(self.value)
            for i, child in enumerate(self.child):
                child_amr = " :ARG{} ".format(i) + child.to_amr(var_exist)
                amr_str += child_amr
            amr_str += ")"
            return amr_str


def parse_lambda(logic_str: str):
    try:
        lg_parent = LogicElement(depth_level=-1)
        tk_arr = logic_str.split()
        tk_arr = [tk for tk in tk_arr if len(tk) > 0]
        tmp_logic = [lg_parent]
        j = 0
        for i in range(len(tk_arr)):
            if i + j >= len(tk_arr):
                break
            tk = tk_arr[i + j]
            if tk == "(":
                if i + j + 1 < len(tk_arr) and not tk_arr[i + j + 1] == "(":
                    new_lg = LogicElement(value=tk_arr[i + j + 1], depth_level=tmp_logic[-1].depth_level+1)
                    j += 1
                else:
                    new_lg = LogicElement(depth_level=tmp_logic[-1].depth_level+1)
                tmp_logic[-1].add_child(new_lg)
                tmp_logic.append(new_lg)
            elif tk == ")":
                tmp_logic.pop()
            else:
                tmp_logic[-1].add_child(LogicElement(value=tk, depth_level=tmp_logic[-1].depth_level+1))
    except:
        print("[exception] parsing logic: {}".format(logic_str))
    return lg_parent


def parse_prolog(logic_str: str):
    lg_parent = LogicElement()
    tk_arr = logic_str.split()
    tk_arr = [tk for tk in tk_arr if tk != "," and len(tk) > 0]
    tmp_logic = [lg_parent]
    for i in range(len(tk_arr)):
        tk = tk_arr[i]
        if tk == "(":
            if i > 0 and (tk_arr[i - 1] == "(" or tk_arr[i - 1] == ")"):
                new_lg = LogicElement()
                tmp_logic[-1].add_child(new_lg)
                tmp_logic.append(new_lg)
            pass
        elif tk == ")":
            tmp_logic.pop()
        else:
            if i + 1 < len(tk_arr) and tk_arr[i + 1] == "(":
                new_lg = LogicElement(value=tk_arr[i])
                tmp_logic[-1].add_child(new_lg)
                tmp_logic.append(new_lg)
            else:
                tmp_logic[-1].add_child(LogicElement(value=tk))

    return lg_parent


if __name__ == "__main__":
    logic_s = "job ( ANS  ) , job ( ANS  ) , salary_greater_than ( ANS , num_salary , year ) , language ( ANS , languageid0 )"
    logic_s2 = "salary_greater_than ( ANS , num_salary , year ) , job ( ANS ) ,  language ( ANS , languageid0 ) language ( ANS , languageid0 ) salary_greater_than ( ANS , num_salary  ) "
    # logic_s = "( call SW.listValue ( call SW.getProperty ( ( lambda s ( call SW.superlative ( var s ) ( string max ) ( call SW.ensureNumericProperty ( string num_rebounds ) ) ) ) ( call SW.domain ( string player ) ) ) ( string player ) ) )"
    # logic_s = "( lambda ?x exist ?y ( and ( mso:@@ wine.@@ wine.@@ gra@@ pe_@@ vari@@ ety 2005 _ joseph _ car@@ r _ nap@@ a _ valley _ ca@@ ber@@ net _ s@@ au@@ vi@@ gn@@ on ?y ) ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composi@@ tion.@@ gra@@ pe_@@ vari@@ ety ?y pe@@ ti@@ t _ ver@@ do@@ t ) ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composition@@ .per@@ cent@@ age ?y ?x ) ) )"
    logic_s2 = "( lambda ?x exist ?y ( and ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composi@@ tion.@@ gra@@ pe_@@ vari@@ ety ?y pe@@ ti@@ t _ ver@@ do@@ t ) ( mso:@@ wine.@@ wine.@@ gra@@ pe_@@ vari@@ ety 2005 _ joseph _ car@@ r _ nap@@ a _ valley _ ca@@ ber@@ net _ s@@ au@@ vi@@ gn@@ on ?y ) ( mso:@@ wine.@@ gra@@ pe_@@ vari@@ e@@ ty_@@ composition@@ .per@@ cent@@ age ?y ?x ) ) )"
    # logic_s = "( count $0 ( and ( state:t $0 ) ( exists $1 ( and ( place:t $1 ) ( loc:t $1 $0 ) ( > ( elevation:i $1 ) ( elevation:i ( argmax $2 ( and ( place:t $2 ) ( exists $3 ( and ( loc:t $2 $3 ) ( state:t $3 ) ( loc:t $3 co0 ) ( loc:t ( argmax $4 ( and ( capital:t $4 ) ( city:t $4 ) ) ( size:i $4 ) ) $3 ( size:i $4 ) ) ) ) ) ( elevation:i $2 ) ) ) ) ) ) ) )"
    logic_s2 = "( count $0 ( and ( state:t $0 ) ( exists $1 ( and  ( loc:t $1 $0 ) ( place:t $1 ) ( > ( elevation:i $1 ) ( elevation:i ( argmax $2 ( and ( place:t $2 ) ( exists $3 ( and ( loc:t $2 $3 ) ( state:t $3 ) ( loc:t $3 co0 ) ( loc:t ( argmax $4 ( and ( capital:t $4 ) ( city:t $4 ) ) ( size:i $4 ) ) $3 ( size:i $4 ) ) ) ) ) ( elevation:i $2 ) ) ) ) ) ) ) )"
    logic_s2 = "( ROOT ( S ( SBAR ( WHNP ( ( WDT which ) ) ) ) ( NP ( NNS airline ) ) ( VP ( VBP serve ) ( NP ( NN ci0 ) ) ) ) )"
    logic_s2 = "( ROOT ( NUR ( S ( PRON es ) ( AUX ist ) ( NP ( PRON diese ) ( NOUN pyrami@@_de ) ) ) ( PUNCT . ) ) )"
    # logic_s2 = """( NP ( N B@@_ong ) ( A n@@_ứt ) ( V là ) ( P gì ) ( ? ? ) ( NP ( N Bạn ) ) ( V thấy ) ( S ( VP ( P đó ) ( E trên ) ( Nc con ) ( N đường ) ( P này ) ( N đá ) ( N lớp ) ( N mặt ) ( V bị ) ( V bong ) ( A tr@@_óc ) ) ) ( . . ) )"""

    from nltk.tree import Tree
    tr= Tree.fromstring(logic_s2)
    print(tr)

    # logic_s2 = '( ROOT ( NP ( XX 13@@_6@@_9 ) ( . . ) ) ( NP hihe ) )'
    s2 = parse_lambda(logic_s2) 
    max_nodes_pruned = s2.get_leaf_nodes_with_depth(3)
    print(" ".join([str(x) for x in max_nodes_pruned]))
    print(" ".join( [e.value.replace("@@_", "@@ ") for e in max_nodes_pruned]))
    # print(s2)
    # s2.flag_vp_in_subtree()
    # s2.prune_tag_novp_in_subtree(["NP", "VP"])

    nodes_pruned = s2.get_leaf_nodes_template()
    print(" ".join( [e.value.replace("@@_", "@@ ") for e in nodes_pruned]))

    print(s2.get_all_node_name())
    # leaf = s2.get_leaf_nodes_with_depth(depth=4)
    # print("\n".join([str(x) for x in leaf]))
    print(s2.get_min_depth())
    print(s2.get_max_depth())
    print(s2.get_max_depth())
    # from template_generator import generate_template
    # import json
    # generate_template({"bpe_template": logic_s2}, json.load(open('../data/train.en.tagfreq.json')))