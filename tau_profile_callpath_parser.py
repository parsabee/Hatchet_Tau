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
        return {"name": self._name, "metrics": self._metrics}
    
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
        Hatchet expects inner nodes of the tree to contain, on top of what a regular node contains,
        one extra field called `children`.children is a list of nodes.
        """
        if len(self._children) == 0:
            return super(InnerNode, self).to_dict()
        
        children = []
        for child in self._children:
            children.append(child.to_dict())
        return {"name": self._name, "metrics": self._metrics, "children": children}
    
    def add_children(self, node):
        self._children.add(node)
        

class CallPaths():
    """
    Generates call paths that are understood by hatchet
    """
    def __init__(self, non_call_path_data, call_path_data):
        """ initializer should not be directly called instead use factory methods """
        self._roots = []
        self._non_call_path = non_call_path_data
        self._call_path_data = call_path_data
        self._depth = len(call_path_data.index.levshape)
        self._an = None
        self._recursive_constructor(self._call_path_data, 0, None)
        
    def get_roots(self):
        """
        creates a json-like (list of dictionaries) object that is understood by hatchet.GraphFrame.from_literal()
        """
        return [root.to_dict() for root in self._roots]
    
    def to_hatchet(self):
        """creates a hatchet graphframe"""
        gf = ht.GraphFrame.from_literal(self.get_roots())
        gf.update_inclusive_columns() # to calculate all of the inclusives 
        return gf
    
    def _make_node(self, name, row, parent_node):
        """creates a node in the callpath tree"""
        
        metrics = {"time": row['Exclusive'],
                   "calls": row['Calls'],
                   "subcalls": row['Subcalls'],
                  }

        node = InnerNode(name, **metrics)
        if parent_node is not None:
            parent_node.add_children(node)
        else:
            self._roots.append(node)
        return node
        
    def _is_node_leaf(self, call_path_data, func, level):
        """checks whether `func` is a leaf, either is the last level or
        it's next level's index is `nan`"""
        if level == self._depth - 1:
            return True
        df = call_path_data.loc[func]
        leaves = df.loc[df.index.get_level_values(0).isnull()]
        return not leaves.empty
                    
    def _recursive_constructor(self, call_path_data, level, parent_node):
        """recursively builds the tree"""
        if level == self._depth:
            return
        
        functions_on_this_level = list(call_path_data.groupby(level=0).groups.keys())
        for func in functions_on_this_level:
            if self._is_node_leaf(call_path_data, func, level):
                row = call_path_data.loc[func]
                if type(row) == pd.core.frame.DataFrame: # when it's not a Series, there are still more levels
                    node = self._make_node(func.strip(), row.loc[np.nan], parent_node)
                    self._recursive_constructor(row.loc[~row.index.isin([np.nan])], level + 1, node)
                else:
                    node = self._make_node(func.strip(), row, parent_node)

            else:
                row = self._non_call_path.loc[func.strip()]
                node = self._make_node(func.strip(), row, parent_node)
                self._recursive_constructor(call_path_data.loc[func], level + 1, node)

    @staticmethod
    def _get_call_paths(data, node, context, thread):
        """extracts callpath data from tau profile data `data`"""
        data = data.loc[node, context, thread]
        data = data[data['Group'].str.contains('TAU_CALLPATH', regex=False)]
        data = data.rename(lambda x: x.strip())
        data = data.set_index(data.index.str.split("\s*=>\s*", expand=True))
        return data
        
    @staticmethod
    def _get_non_call_paths(data, node, context, thread):
        """extracts every group but the callpaths from tau profile data `data`"""
        data = data.loc[node, context, thread]
        data = data[~data['Group'].str.contains('TAU_CALLPATH', regex=False)]
        data = data.set_index(data.index.str.strip())
        return data

    @staticmethod
    def from_tau_interval_profile(tau_interval, node, context, thread):
        """
        Creates and returns a CallPath object
        
        tau_interval: pandas.DataFrame; the interval data from TauProfileParser
        node: int
        contex: int
        thread: int
        """
        non_call_path_data = CallPaths._get_non_call_paths(tau_interval, node, context, thread)
        call_path_data = CallPaths._get_call_paths(tau_interval, node, context, thread)
        return CallPaths(non_call_path_data, call_path_data)
        