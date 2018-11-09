# -*- coding: utf-8 -*-

class CatBoostCategoricalModel(object):

	def __init__(self, CatboostModel):
		self.model = CatboostModel

		self.init_tree_origin_info()
		self.init_tree_info()

		self.highlight_params = {
			'color': "coral2",
			'fillcolor': "coral2", 
			'style': 'filled'
		}

		self.binary_features = None

	def init_tree_origin_info(self):
		tree_origin_info = []

		tree_splits_index = 0
		current_tree_leaf_values_index = 0
		for index in range(self.model.tree_count):
			current_tree_depth = self.model.tree_depth[index]

			feature_list = self.model.tree_split_feature_index[tree_splits_index: tree_splits_index + current_tree_depth]
			xor_mask_list = self.model.tree_split_xor_mask[tree_splits_index: tree_splits_index + current_tree_depth]
			border_val_list = self.model.tree_split_border[tree_splits_index: tree_splits_index + current_tree_depth]
			tree_leaf_values = self.model.leaf_values[current_tree_leaf_values_index: current_tree_leaf_values_index + (1 << current_tree_depth)]

			tree_splits_index += current_tree_depth
			current_tree_leaf_values_index += (1 << current_tree_depth)

			tree_origin_info.append({'current_tree_depth': current_tree_depth,
				'feature_list': feature_list, 
				'xor_mask_list': xor_mask_list, 
				'border_val_list': border_val_list, 
				'tree_leaf_values': tree_leaf_values
			})

		self.tree_origin_info = tree_origin_info

	def init_tree_info(self):

		def build_node(depth, index, feature_index, xor_mask, border):
			node = {}
			node['key'] = '%d_%d' % (depth, index)
			node['feature_index'] = feature_index
			node['xor_mask'] = xor_mask
			node['border'] = border
			if xor_mask != 0:
				node['label'] = '(feature_%d^%d)>=%f' % (feature_index, xor_mask, border)
			else:
				node['label'] = 'feature_%d>=%f' % (feature_index, border)
			node['left'] = {}
			node['right'] = {}
			tree_node_index[node['key']] = node
			return node

		def build_leaf(depth, index, values):
			leaf = {}
			leaf['key'] = '%d_%d' % (depth, index)
			leaf['value'] = values
			leaf['label'] = str(values)
			tree_node_index[leaf['key']] = leaf
			return leaf

		def build_edge(depth, index, node):
			edge_key = '%d_%d' % (depth-1, index // 2)
			direction = 'left' if index % 2 == 0 else 'right'
			tree_node_index[edge_key][direction] = node

		def set_node_zero(node):
			node['label'] = '0'
			node['value'] = 0

			del node['feature_index']
			del node['xor_mask']
			del node['border']
			del node['left']
			del node['right']

		def pruning():
			while True:
				exit = True
				delete_keys = []
				for key in tree_node_index:
					node = tree_node_index[key]
					if 'left' not in node or 'right' not in node:
						continue
					if node['left']['label'] == '0' and node['right']['label'] == '0':
						delete_keys.append(node['left']['key'])
						delete_keys.append(node['right']['key'])
						set_node_zero(node)
						exit = False
				for key in delete_keys:
					del tree_node_index[key]
				if exit:
					break

		tree_info = []
		for tree_id in range(self.model.tree_count):
			tree_node_index = {}
			current_tree_depth = self.tree_origin_info[tree_id]['current_tree_depth']

			feature_list = self.tree_origin_info[tree_id]['feature_list']
			xor_mask_list = self.tree_origin_info[tree_id]['xor_mask_list']
			border_val_list = self.tree_origin_info[tree_id]['border_val_list']
			tree_leaf_values = self.tree_origin_info[tree_id]['tree_leaf_values']

			if current_tree_depth == 0:
				leaf = build_leaf(current_tree_depth, 0, tree_leaf_values[0])
				tree_info.append({
					'feature_list': None,
					'tree_root': leaf,
					'tree_node_index': None
				})
				continue

			# print current_tree_depth, len(feature_list)
			root_feature_index = feature_list[current_tree_depth - 1]
			root_border = border_val_list[current_tree_depth - 1]
			root_mask = xor_mask_list[current_tree_depth - 1]
			tree_root = build_node(0, 0, root_feature_index, root_mask, root_border)

			for depth in range(1, current_tree_depth):
				feature_index = feature_list[current_tree_depth - 1 - depth]
				xor_mask = xor_mask_list[current_tree_depth - 1 - depth]
				border = border_val_list[current_tree_depth - 1 - depth]
				for j in range(1 << depth):
					node = build_node(depth, j, feature_index, xor_mask, border)
					build_edge(depth, j, node)

			for j in range(1 << current_tree_depth):
				leaf = build_leaf(current_tree_depth, j, tree_leaf_values[j])
				build_edge(current_tree_depth, j, leaf)

			pruning()

			tree_info.append({
				'feature_list':set(feature_list),
				'tree_root': tree_root,
				'tree_node_index': tree_node_index
			})

		self.tree_info = tree_info

	def set_binary_features(self, binary_features):
		self.binary_features = binary_features
		return self

	def get_result(self):
		result = 0
		for tree_id in range(self.model.tree_count):
			tree = self.tree_info[tree_id]['tree_root']
			while 'value' not in tree:
				if (self.binary_features[tree['feature_index']] ^ tree['xor_mask']) >= tree['border']:
					tree = tree['right']
				else:
					tree = tree['left']
			result += tree['value']
		return result

	def draw_tree(self, dot, tree_root):

		def recursion_add(root, parent=None, decision=None, in_road=False):
			if 'feature_index' in root:
				# non-leaf
				if in_road:
					dot.node(root['key'], root['label'], **self.highlight_params)
				else:
					dot.node(root['key'], root['label'])

				new_road = ''
				if self.binary_features is not None and in_road:
					if (self.binary_features[root['feature_index']] ^ root['xor_mask']) >= root['border']:
						new_road = 'right'
					else:
						new_road = 'left'
				
				recursion_add(root['left'], root['key'], 'No', new_road=='left')
				recursion_add(root['right'], root['key'], 'Yes', new_road=='right')
			else:
				# leaf
				if in_road:
					dot.node(root['key'], root['label'], **self.highlight_params)
				else:
					dot.node(root['key'], root['label'])

			if parent is not None:
				dot.edge(parent, root['key'], decision)

		recursion_add(tree_root, in_road=(True if self.binary_features is not None else False))

		return dot

	def draw_sub_tree(self, dot, tree_id):
		self.draw_tree(dot, self.tree_info[tree_id]['tree_root'])

	def draw_all_tree(self, dot):
		for tree_id in range(self.model.tree_count):
			self.draw_tree(dot, self.tree_info[tree_id]['tree_root'])
