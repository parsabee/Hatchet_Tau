"""
The callpath data is in the TAU_CALLPATH group, 
so we filter our dataframe to give us only the callpaths. 
Further, the calls in the callpath are joined by =>, 
so we split them into a hierarchical index, 
and construct a tree from the generated multiindexed dataframe.
"""

from tau_profile_parser import TauProfileParser
import pandas as pd
import numpy as np
import hatchet as ht
import re

class Node():
    """
    Abstract node of a tree that will be passed to hatchet API
    """
    def __init__(self):
        raise NotImplemented("Node is abstract")
        
    def to_dict(self):
        """
        Hatchet expects a dictionary representation of each node with at least
        the keys `name`(string) and `metrics`(dict).
        an examplie of metrics would be : {"inclusive_time": 10.0, "exclusive_time": 9.0}
        """
        return {"name": self._name, "metrics": self._metrics.copy()}
    
    def _initialize(self, name, **metrics):
        """constructor, called by subclasses"""
        self.update_name(name)
        self.update_metrics(**metrics)
        
    def get_name(self):
        return self._name
    
    def get_metrics(self):
        return self._metrics
        
    def update_name(self, name):
        self._name = name
        
    def update_metrics(self, **metrics):
        self._metrics = metrics
            
class InnerNode(Node):
    """A node with children"""
    def __init__(self, name, **kwargs):
        """
        name: str; name of the timer
        kwargs; metrics of the node
        """
        super(InnerNode, self)._initialize(name, **kwargs)
        self._children = set()
    
    def to_dict(self):
        """
        Hatchet expects inner nodes of the tree to contain `children`. `children' is a list of nodes.
        """
        if len(self._children) == 0:
            return super(InnerNode, self).to_dict()
        
        children = [child.to_dict() for child in self._children]
        return {"name": self._name, "metrics": self._metrics.copy(), "children": children.copy()}
    
    def add_children(self, node):
        self._children.add(node)
        

class CallPaths():
    """
    Generates call paths that are understood by hatchet
    """
    
    @staticmethod
    def _make_node(name, row, parent_node, roots, nodes):
        """creates a node in the callpath tree"""
        if parent_node:
            key = f"{name}:{parent_node.get_name()}"
        else:
            key = f"{name}:root"
            
        if key in nodes:
            return nodes[key]
        
        metrics = {
            "time": row['Exclusive'],
            "time (inc)": row['Inclusive'],
            "calls": row['Calls'],
            "subcalls": row['Subcalls'],
            "groups": row['Group'][7:-1] #getting rid of the 'Group="' and the '"' at the beginning and end of Group string
        }

        node = InnerNode(name, **metrics)  
        if parent_node is not None:
            parent_node.add_children(node)
        else:
            roots.append(node)
            
        if nodes:
            nodes[key] = node
        return node
    
    @staticmethod
    def _is_node_leaf(call_path_data, func, level, depth):
        """checks whether `func` is a leaf, either is the last level or
        it's next level's index is `nan`"""
        if level == depth - 1:
            return True
        
        df = call_path_data.loc[func]
        leaves = df.loc[df.index.get_level_values(0).isnull()]
        return not leaves.empty
    
    @staticmethod
    def _get_last_null_index(row):
        try:
            index = [np.nan] * len(row.index.levshape)
            return row.loc[tuple(index)]
        except AttributeError:
            return row.loc[np.nan]
    
    @staticmethod
    def _recursive_constructor(call_path_data, non_call_path_data, level, parent_node, roots, nodes, depth, drop_context):
        """recursively builds the tree"""
        if level == depth:
            return
        
        functions_on_this_level = list(call_path_data.groupby(level=0).groups.keys())
        for func in functions_on_this_level:
            if CallPaths._is_node_leaf(call_path_data, func, level, depth):
                row = call_path_data.loc[func]
                if type(row) == pd.core.frame.DataFrame: # when it's not a Series, there are still more levels
                    node = CallPaths._make_node(func, CallPaths._get_last_null_index(row), parent_node, roots, nodes)
                    CallPaths._recursive_constructor(row, non_call_path_data, level + 1, node, roots, nodes, depth,drop_context)
                else:
                    node = CallPaths._make_node(func, row, parent_node, roots, nodes)
            else:
                row = call_path_data.loc[func]
                if drop_context and "[CONTEXT]" in func:
                    node = parent_node
                else:
                    try:
                        node = CallPaths._make_node(func, CallPaths._get_last_null_index(row), parent_node, roots, nodes)
                    except KeyError:
                        # can't make node, we don't have info for it's timer, 
                        # paraprof uses the flat profile in this case
                        node = CallPaths._make_node(func, non_call_path_data.loc[func], parent_node, roots, nodes)
                
                CallPaths._recursive_constructor(call_path_data.loc[func], non_call_path_data, level + 1, node, roots, nodes, depth, drop_context)
    
    @staticmethod
    def _construct(call_path_data, non_call_path_data, drop_context):
        depth = len(call_path_data.index.levshape)
        nodes = dict()
        roots = []
        CallPaths._recursive_constructor(call_path_data, non_call_path_data, 0, None, roots, nodes, depth, drop_context)
        roots = [root.to_dict() for root in roots]
        return CallPaths(roots, call_path_data, non_call_path_data, depth)
        
    def __init__(self, roots, call_paths, non_call_paths, depth):
        """ initializer should not be directly called instead use factory methods """
        self._roots = roots
        self._depth = depth
        self._call_paths = call_paths
        self._non_call_paths = non_call_paths
        
    def get_roots(self):
        """
        creates a json-like (list of dictionaries) object that is understood by hatchet.GraphFrame.from_literal()
        """
        import copy
        return copy.deepcopy(self._roots)
    
    def apply_action_on_roots(self, action):
        def apply(node, action):
            action(node)
            if 'children' in node.keys():
                for ch_node in node['children']:
                    apply(ch_node, action)
        
        roots = self.get_roots()
        for root in roots:
            apply(root, action)
            
        return CallPaths(roots, self._call_paths, self._non_call_paths, self._depth)
        
        
    def get_exclusives_per_call_roots(self):
        """
        """
        def action(node):
            node['metrics']['time'] /= node['metrics']['calls']
            del node['metrics']['calls']# we no longer need 
            del node['metrics']['time (inc)']
            del node['metrics']['subcalls']
            
        return self.apply_action_on_roots(action)
    
    def to_hatchet(self):
        return ht.GraphFrame.from_literal(self._roots)
    
    def to_sunburst_chart(self, by, top_n=10):
        
        # values to be filled and passed to plotly
        parents = []
        children = []
        values = []
            
        def fillParentsChildren(roots, parent=''):
            for i in roots:
                children.append(i['name'])
                parents.append(parent)
                values.append(i['metrics']['time'])
                if ('children' in i.keys()):
                    fillParentsChildren(i['children'], i['name'])
        
        fillParentsChildren(self._roots)
        
        import plotly.graph_objects as go

        fig =go.Figure(go.Sunburst(
            labels=children,
            parents=parents,
            values=values,
        ))
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
        fig.show()

    @staticmethod
    def to_heat_map(data, metric_name, node, context, threads='all', drop_context=True):
        """Available metric_names: 'Exclusive', 'Inclusive', 'Calls', 'Subcalls' """
        
        import plotly.graph_objs as go
        import plotly as py
        import plotly.express as px
        def get_common_calls_amongst_threads(data, col_name, node, context, threads):
            """
            creates and returns a dataframe containing the `top_n` `col_name` column of the `data` DataFrame
            """
            
            df = None
            cntr = 0
            first = True
            for thread in threads:
                tmp = data.loc[node, context, thread][[col_name]] 
                tmp = tmp.sort_values(by=[col_name], ascending=False)  
                if first: 
                    df = tmp
                else: 
                    df = df.merge(tmp, how='inner', on=['Timer'], suffixes=(f"_{str(cntr)}", f"_{str(cntr + 1)}"))
                    cntr += 1
                first = False
                
            return df

        def create_heat_map_data(data): 
            """Creates a 2d list which is the `heat` data needed for the heat map"""
            heat_map_data = []
            for r in data.iterrows():
                row = [i for i in r[1]]
                heat_map_data.append(row)
            return heat_map_data
        
        if threads == 'all':
            threads = [i for i in range(len(data.groupby(level=2)))]
        if drop_context:
            data = data[~data['Group'].str.contains('TAU_SAMPLE_CONTEXT')]
        thread_labels = [f"thrd_{str(t)}" for t in threads]
        data= get_common_calls_amongst_threads(data, metric_name, node, context, threads)
        function_labels = [f for f in data.index]
        heat_map_data = create_heat_map_data(data)
        fig = go.Figure(data=go.Heatmap(
                        z=heat_map_data,
                        x=thread_labels,
                        y=function_labels
                    ))
        fig.show()

    @staticmethod
    def _categorize_data(data, node, context, thread):
        """extracts callpath data from tau profile data `data`"""
        data = data.loc[node, context, thread]
        data1 = data[data['Group'].str.contains('TAU_CALLPATH', regex=False)]
        data1 = data1.rename(lambda x: x.strip())
        data1 = data1.set_index(data1.index.str.split("\s*=>\s*", expand=True))
        data2 = data[~data['Group'].str.contains('TAU_CALLPATH', regex=False)]
        return data1, data2

    @staticmethod
    def from_tau_interval_profile(tau_interval, node, context, thread, drop_context=True):
        """
        Creates and returns a CallPath object
        
        tau_interval: pandas.DataFrame; the interval data from TauProfileParser
        node: int
        contex: int
        thread: int
        """
        if drop_context:
            tau_interval = tau_interval[~tau_interval['Group'].str.contains('TAU_SAMPLE_CONTEXT')]
            
        call_path_data, non_call_path_data = CallPaths._categorize_data(tau_interval, node, context, thread)
        return CallPaths._construct(call_path_data, non_call_path_data, drop_context)
        