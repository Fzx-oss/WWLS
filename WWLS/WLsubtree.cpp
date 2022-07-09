#include <random>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include "./pybind11/pybind11.h"
#include "./pybind11/stl.h"

namespace py = pybind11;

std::random_device rd;



class WLsubtree {
private:
  // node information : id and label
  struct Node_info{
    int id;
    unsigned long long label;
    Node_info(int id, unsigned long long label): id(id), label(label) {;}
  };

  // node with id, label, and children
  struct Node {
    int id;
    unsigned long long label;
    int depth;
    unsigned long long hash_value1 = 1;
    unsigned long long hash_value2 = 1;
    std::vector<Node*> children;
    Node(int id, unsigned long long label, int depth): id(id), label(label), depth(depth){;}
  };

  // variables
  std::unordered_map<int, std::vector<Node_info>> topo_info; // [node id : adjacent node id]
  std::unordered_map<int, int> memo_CST; // [complete subtree id : number of trees]
  std::unordered_map<std::string, int> memo_All; // memorize all complete subtrees
  std::vector<unsigned long long> random_nums1_1; // vector to store random numbers
  std::vector<unsigned long long> random_nums1_2; // vector to store random numbers
  std::vector<unsigned long long> random_nums2_1; // vector to store random numbers
  std::vector<unsigned long long> random_nums2_2; // vector to store random numbers
  int max_depth; // max iterations
  unsigned long long mod; // max random number
  Node* root = NULL; // root of the WL subtree
  int num_CST; // number of all complete subtrees
  int new_label = -1;


public:
  // constructor
  WLsubtree(int max_iter = 3, unsigned long long mod = 1000000007): max_depth(max_iter), mod(mod){
    // initialize random numbers
    for (int i = 0; i < max_iter; i++) {
      random_nums1_1.emplace_back(((unsigned long long)rand()) % mod);

    }
    for (int i = 0; i < max_iter; i++) {
      random_nums1_2.emplace_back(((unsigned long long)rand()) % mod);
    }
    for (int i = 0; i < max_iter; i++) {
      random_nums2_1.emplace_back(((unsigned long long)rand()) % mod);
    }
    for (int i = 0; i < max_iter; i++) {
      random_nums2_2.emplace_back(((unsigned long long)rand()) % mod);
    }

    // for (int i = 0; i < max_iter; i++) {
    //   std::cout << random_nums[i] << std::endl;
    // }
  }


  // input graph information and initialize topoG
  void input_graph(std::unordered_map<int, std::vector<int>> adj_info, // adjInfo: adjacent node information
                    std::unordered_map<int, int> att_info) {          // attInfo: node label information
    for (auto itr = adj_info.begin(); itr != adj_info.end(); itr++) {
      for (int id : itr->second) { // the id of the adjacent node
        Node_info node_info(id, att_info[id]); 
        topo_info[itr->first].emplace_back(node_info); // [node id : adjacent node info (id and label)]
      }
    }
  }


private:
  // relabel the hash value
  int get_next_label() {
    new_label += 1;
    return new_label;
  }


  // function that recursively adds nodes to build a WL subtree
  Node* add_node(int id, unsigned long long label, int depth) {
    if (depth > max_depth) {
      return NULL;
    }

    // generate a new node
    Node* new_node = new Node(id, label, depth);

    // recursively add each node
    for (Node_info target : topo_info[id]) {
      Node* child = add_node(target.id, target.label, depth+1);
      if(child != NULL) {
        new_node->children.emplace_back(child);
      }
    }

    // calculate the hash value1 of each complete subtree
    if (depth != max_depth) {
      // Hash 1
      for (Node* child_node : new_node->children) {
        new_node->hash_value1 = (new_node->hash_value1 *
                                ((child_node->hash_value1 + random_nums1_1[new_node->depth]) % mod)
                                ) % mod; // node not a leaf
        // delete childNode; // free this child node
      }
      if (new_node->children.size() == 0) {
        new_node->hash_value1 = new_node->label + 1; // leaf  
      } else {
        new_node->hash_value1 = ((new_node->label + 1 + random_nums1_2[new_node->depth]) 
                                * new_node->hash_value1) % mod; // not leaf
      }

      // Hash 2
      for (Node* child_node : new_node->children) {
        new_node->hash_value2 = (new_node->hash_value2 *
                                ((child_node->hash_value2 + random_nums2_1[new_node->depth]) % mod)
                                ) % mod; // node not a leaf
        // delete childNode; // free this child node
      }
      if (new_node->children.size() == 0) {
        new_node->hash_value2 = new_node->label + 1; // leaf  
      } else {
        new_node->hash_value2 = ((new_node->label + 1 + random_nums2_2[new_node->depth]) 
                                * new_node->hash_value2) % mod; // not leaf
      }
      

    } else {
      // Hash 1
      new_node->hash_value1 = new_node->label + 1; // leaf
      // Hash 2
      new_node->hash_value2 = new_node->label + 1;
    }

    std::string new_hash_str = std::to_string(new_node->hash_value1) + std::to_string(new_node->hash_value2);

    // record all types of complete subtrees & relabel the hash value of the cossesponding WL subtree
    if (memo_All.find(new_hash_str) == memo_All.end()) {
      memo_All[new_hash_str] = get_next_label();
    }

    // count the number of each complete subtree
    memo_CST[memo_All[new_hash_str]] += 1;

    return new_node;
  }



public:
  // function to build a WL subtree
  void build_WLsubtree(int root_id, int root_label) {
    root = add_node(root_id, root_label, 0);
    num_CST = memo_All.size();
  }


  // function to return a unordered_map that contains node feature
  std::unordered_map<int, int> get_feature() {
    return memo_CST;
  }


  // function to return the number of all kinds of complete subtrees
  int get_num_CST() {
    return num_CST;
  }

  // function to free memory
  void clear(bool all = false) {
    topo_info.clear();
    memo_CST.clear();
    root = NULL;
    if (all == true) {
      memo_All.clear();
      random_nums1_1.clear();
      random_nums1_2.clear();
      random_nums2_1.clear();
      random_nums2_2.clear();
      num_CST = 0;
    }
  }


};



// Python interface
PYBIND11_MODULE(WLsubtree, m) {
  py::class_<WLsubtree>(m, "WLsubtree")
      .def(py::init<int, long long>(), py::arg("max_iter")=2, py::arg("mod")=1000000007)
      .def("input_graph", &WLsubtree::input_graph, 
            "A function to input graph information", py::arg("adj_info"), py::arg("att_info"))
      .def("build_WLsubtree", &WLsubtree::build_WLsubtree, py::arg("root_id"), py::arg("root_label"))
      .def("get_feature", &WLsubtree::get_feature, "A function to return dictionary that contains node feature")
      .def("get_num_CST", &WLsubtree::get_num_CST, "A function to return the number of all kinds of complete subtrees")
      .def("clear", &WLsubtree::clear, "A funciton to free memory", py::arg("all")=false);
}
