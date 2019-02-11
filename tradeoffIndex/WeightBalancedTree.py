def subTreeSize(subTreeRoot):
    if subTreeRoot is None: return 0
    else: return subTreeRoot.size

# class basicBinIntTree(): # TODO: turn into a weight-balanced tree to have some notion of efficiency
#     def __init__(self, rootElem, leftSubTree=None, rightSubTree=None):
#         self.rootElem=rootElem
#         self.leftSubTree=leftSubTree
#         self.rightSubTree=rightSubTree
#         self.numel=len(leftSubTree)+len(rightSubTree)+1
#
#     def __len__(self):
#         return self.numel # should control access to subtrees to prevent from becoming inaccurate
#
#     @classmethod
#     def fromSortedList(cls, list):
#         leftPreceeding=bi.bisect_left(list)
#         rightPreceeding=bi.bisect_right(list)
#         leftSubTree=basicBinIntTree.fromSortedList(list[0:leftPreceeding])
#         rightSubTree=basicBinIntTree.fromSortedList(list[rightPreceeding:len(list)])
#         return basicBinIntTree(list[leftPreceeding:rightPreceeding], leftSubTree, rightSubTree)
#         # post-process and acumulate returns
#
#     @classmethod
#     def fromIter(cls, list, cmpKey=None):
#         return basicBinIntTree.fromSortedList(sorted(list), cmpKey=cmpKey)
#
#     def __contains__(self, item):
#         if item in self.rootElems: return True
#         if self.leftSubTree is not None and item in self.leftSubTree: return True
#         if self.rightSubTree is not None and item in self.rightSubTree: return True
#         return False

class WeightBalancedTreeNode():
    def __init__(self,key, value=None, leftSubTree=None, rightSubTree=None):
        self.key=key
        self.value=value
        self.left=leftSubTree
        self.right=rightSubTree
        self.size=subTreeSize(leftSubTree)+subTreeSize(rightSubTree)

    @property
    def weight(self):
        return self.size+1

    def _rotate(self, predecessor, isRotLeft):
        if isRotLeft:
