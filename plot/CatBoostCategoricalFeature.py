# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import product
from functools import reduce


class CatBoostCategoricalFeature():

	def __init__(self, CatboostModel, feaure_info):
		self.model = CatboostModel.catboost_model
		self.hash_uint64 = CatboostModel.hash_uint64
		self.calc_hash = CatboostModel.calc_hash

		self.float_feature_name = [feaure_info.float_feature_name[i] for i in self.model.float_features_index]
		self.float_feature_name_count = len(self.float_feature_name)
		self._discrete_float_feature()

		self.categorical_feature_name = feaure_info.categorical_feature_name
		self._one_hot_feature()
		self.one_hot_feature_count = len(self.one_hot_feature)

		self._calc_ctrs()
		self.ctrs_feature_count = sum([len(ctrs['generate_action']) for ctrs in self.interaction_ctrs])

		self.all_feature_count = self.float_feature_name_count + self.one_hot_feature_count + self.ctrs_feature_count

		self.original_feature = None
		self.binary_features = None

	def _discrete_float_feature(self):
		discrete_float_feature = defaultdict(dict)
		for index, feature_name in enumerate(self.float_feature_name):
			discrete_float_feature[feature_name] = self.model.float_feature_borders[index]
		self.discrete_float_feature = discrete_float_feature

	def _one_hot_feature(self):

		one_hot_feature = []

		cat_feature_packed_indexes = {}
		for i in range(self.model.cat_feature_count):
			cat_feature_packed_indexes[self.model.cat_features_index[i]] = i
		
		for i in range(len(self.model.one_hot_cat_feature_index)):
			cat_idx = cat_feature_packed_indexes[self.model.one_hot_cat_feature_index[i]]
			feature_name = self.categorical_feature_name[cat_idx]
			one_hot_feature.append(feature_name)

		self.one_hot_feature = one_hot_feature

	def _calc_ctrs(self):
		model_ctrs = self.model.model_ctrs
		interaction_ctrs = []
		border_index = 0
		for i in range(len(model_ctrs.compressed_model_ctrs)):
			proj = model_ctrs.compressed_model_ctrs[i].projection
			category_features = [self.categorical_feature_name[cat_feature_index] for cat_feature_index in proj.transposed_cat_feature_indexes]

			bin_features = []
			bin_featrue_rules = defaultdict(dict)
			for _, bin_feature_index in enumerate(proj.binarized_indexes):
				if bin_feature_index.bin_index < self.float_feature_name_count:
					feature_name = self.float_feature_name[bin_feature_index.bin_index]
					bin_featrue_rules[feature_name]['value'] = self.discrete_float_feature[feature_name][bin_feature_index.value-1]
					bin_featrue_rules[feature_name]['type'] = 'equal' if bin_feature_index.check_value_equal else 'compare'
				else:
					feature_name = self.one_hot_feature[bin_feature_index.bin_index - self.float_feature_name_count]
					bin_featrue_rules[feature_name]['value'] = bin_feature_index.value
					bin_featrue_rules[feature_name]['type'] = 'oh_equal' if bin_feature_index.check_value_equal else 'oh_compare'
					bin_featrue_rules[feature_name]['index'] = bin_feature_index.bin_index - self.float_feature_name_count
				
				bin_features.append(feature_name)
				
				

			ctrs_feature = {'category_features': category_features, 'float_features': bin_features, 'float_featrue_rules': bin_featrue_rules}

			generate_action = []
			for j in range(len(model_ctrs.compressed_model_ctrs[i].model_ctrs)):
				ctr = model_ctrs.compressed_model_ctrs[i].model_ctrs[j]
				learn_ctr = model_ctrs.ctr_data.learn_ctrs[ctr.base_hash]
				generate_action.append({'ctr': ctr, 'learn_ctr': learn_ctr, 'borders':  self.model.ctr_feature_borders[border_index]})
				border_index += 1

			interaction_ctrs.append({'ctrs_feature': ctrs_feature, 'generate_action': generate_action})

		self.interaction_ctrs = interaction_ctrs

	def _deal_float_features(self, float_features):
		result = [0] * self.float_feature_name_count
		for index, _ in enumerate(float_features):
			borders = self.discrete_float_feature[self.float_feature_name[index]]
			for _, border in enumerate(borders):
				result[index] += 1 if float_features[index] > border else 0
		return result

	def _calc_one_hot_feature(self, index, feature_value):
		one_hot_hash_values = self.model.one_hot_hash_values[index]
		hash_value = self.hash_uint64(feature_value)
		binary_features = 0
		for border_idx in range(len(one_hot_hash_values)):
			binary_features |= (1 if hash_value == one_hot_hash_values[border_idx] else 0) * (border_idx + 1)
		return binary_features

	def _deal_one_hot_feature(self, cat_features):
		result = [0] * self.one_hot_feature_count
		for index, feature in enumerate(self.one_hot_feature):
			value_index = self.categorical_feature_name.index(feature)
			result[index] = self._calc_one_hot_feature(index, cat_features[value_index])
		return result

	def _apply_action(self, ctr, learn_ctr, ctr_hash):
		result = 0
		ctr_type = ctr.base_ctr_type
		bucket = learn_ctr.resolve_hash_index(ctr_hash)
		if bucket is None:
			result = ctr.calc(0, 0)
		else:
			if ctr_type == "BinarizedTargetMeanValue" or ctr_type == "FloatTargetMeanValue":
				ctr_mean_history = learn_ctr.ctr_mean_history[bucket]
				result = ctr.calc(ctr_mean_history.sum, ctr_mean_history.count)
			elif ctr_type == "Counter" or ctr_type == "FeatureFreq":
				ctr_total = learn_ctr.ctr_total
				denominator = learn_ctr.counter_denominator
				result = ctr.calc(ctr_total[bucket], denominator)
			elif ctr_type == "Buckets":
				ctr_history = learn_ctr.ctr_total
				target_classes_count = learn_ctr.target_classes_count
				total_count = 0
				good_count = ctr_history[bucket * target_classes_count + ctr.target_border_idx]
				for class_id in range(target_classes_count):
					total_count += ctr_history[bucket * target_classes_count + class_id]
				result = ctr.calc(good_count, total_count)
			else:
				ctr_history = learn_ctr.ctr_total
				target_classes_count = learn_ctr.target_classes_count

				if target_classes_count > 2:
					good_count = 0
					total_count = 0
					for class_id in range(ctr.target_border_idx + 1):
						total_count += ctr_history[bucket * target_classes_count + class_id]
					for class_id in range(ctr.target_border_idx + 1, target_classes_count):
						good_count += ctr_history[bucket * target_classes_count + class_id]
					total_count += good_count
					result = ctr.calc(good_count, total_count)
				else:
					result = ctr.calc(ctr_history[bucket * 2 + 1], ctr_history[bucket * 2] + ctr_history[bucket * 2 + 1])
			return result

	def _deal_ctrs_feature(self, float_features, cat_features):
		result = [0] * self.ctrs_feature_count
		result_index = 0
		for interaction_ctr in self.interaction_ctrs:
			ctrs_feature = interaction_ctr['ctrs_feature']
			float_featrue_rules = ctrs_feature['float_featrue_rules']
			temp_value = [0] * (len(ctrs_feature['category_features']) + len(ctrs_feature['float_features']))
			temp_index = 0
			for feature_name in ctrs_feature['category_features']:
				temp_value[temp_index] = self.hash_uint64(cat_features[self.categorical_feature_name.index(feature_name)])
				temp_index += 1
			for feature_name in ctrs_feature['float_features']:
				rule = float_featrue_rules[feature_name]
				value = 0
				if feature_name in self.float_feature_name:
					value = float_features[self.float_feature_name.index(feature_name)]
				else:
					value = cat_features[self.categorical_feature_name.index(feature_name)]
				
				if rule['type'] == 'compare':
					temp_value[temp_index] = 1 if value > rule['value'] else 0
				elif rule['type'] == 'equal':
					temp_value[temp_index] = 1 if value == rule['value'] else 0
				elif rule['type'] == 'oh_compare':
					temp_value[temp_index] = 1 if self._calc_one_hot_feature(rule['index'], value) > rule['value'] else 0
				else: # rule['type'] == 'oh_equal'
					temp_value[temp_index] = 1 if self._calc_one_hot_feature(rule['index'], value) == rule['value'] else 0
				temp_index += 1
			ctr_hash = reduce(lambda x, y: self.calc_hash(x, y), temp_value, 0)
			for action in interaction_ctr['generate_action']:
				ctrs = self._apply_action(action['ctr'], action['learn_ctr'], ctr_hash)
				for border in action['borders']:
					result[result_index] += 1 if ctrs > border else 0
				result_index += 1

		return result

	def deal_orginal_feature(self, float_features, cat_features):

		float_features = [float_features[i] for i in self.model.float_features_index]

		float_result = self._deal_float_features(float_features)
		one_hot_result = self._deal_one_hot_feature(cat_features)
		ctrs_feature_result = self._deal_ctrs_feature(float_features, cat_features)
		return float_result + one_hot_result + ctrs_feature_result
	
	def reset_feature_value(self):
		self.original_feature = None
		self.binary_features = None

	def set_feature_value(self, float_features, cat_features):
		binary_features = self.deal_orginal_feature(float_features, cat_features)
		original_feature = {}
		for index, name in enumerate(self.float_feature_name):
			original_feature[name] = float_features[index]
		for index, name in enumerate(self.categorical_feature_name):
			original_feature[name] = cat_features[index]

		self.original_feature = original_feature
		self.binary_features = binary_features

		return self.binary_features

	def _draw_float_feature(self, dot, float_features):
		for index in float_features:
			feature_name = self.float_feature_name[index]
			if self.original_feature is None:
				dot.node(feature_name, feature_name)
			else:
				dot.node(feature_name, '%s: %s' % (feature_name, str(self.original_feature[feature_name])))

			disperse_node = 'disperse_%d' % index
			dot.node(disperse_node, 'discrete')
			dot.edge(feature_name, disperse_node, label='')

			feature_node = 'feature_%d' % index
			if self.binary_features is None:
				dot.node(feature_node, feature_node)
			else:
				dot.node(feature_node, '%s: %d' % (feature_node, self.binary_features[index]))
			dot.edge(disperse_node, feature_node, label='')

	def _draw_one_hot_feature(self, dot, cat_features):
		for index in cat_features:
			feature_name = self.one_hot_feature[index - self.float_feature_name_count]
			if self.original_feature is None:
				dot.node(feature_name, feature_name)
			else:
				dot.node(feature_name, '%s: %s' % (feature_name, self.original_feature[feature_name]))

			map_node = 'map_%d' % index
			dot.node(map_node, 'one hot map')
			dot.edge(feature_name, map_node, label='')

			feature_node = 'feature_%d' % index
			if self.binary_features is None:
				dot.node(feature_node, feature_node)
			else:
				dot.node(feature_node, '%s: %d' % (feature_node, self.binary_features[index]))
			dot.edge(map_node, feature_node, label='')

	def _build_feature_ctr_map(self):
		if not hasattr(self, 'feature_ctr_map'):
			feature_ctr_map = {}
			index = self.float_feature_name_count + self.one_hot_feature_count
			for ctrs_index, ctrs in enumerate(self.interaction_ctrs):
				for action_index in range(len(ctrs['generate_action'])):
					feature_ctr_map[index] = '%d_%d' % (ctrs_index, action_index)
					index += 1
			self.feature_ctr_map = feature_ctr_map

	def _draw_ctrs_feature(self, dot, ctrs_features):
		ctrs_features.sort()
		self._build_feature_ctr_map()

		feature_index_dict = defaultdict(dict)
		ctrs_features_dict = defaultdict(list)
		for index in ctrs_features:
			ctr_index, action_index  = map(int, self.feature_ctr_map[index].split('_'))
			ctrs_features_dict[ctr_index].append(action_index)
			feature_index_dict[ctr_index][action_index] = index

		for ctr_index in ctrs_features_dict:

			interaction_ctr = self.interaction_ctrs[ctr_index]
			interaction_node = 'interaction_' + str(ctr_index)
			dot.node(interaction_node, interaction_node)

			ctrs_feature = interaction_ctr['ctrs_feature']

			for feature_name in ctrs_feature['category_features']:
				ctr_node_name = 'interaction_%d_%s' % (ctr_index, feature_name)
				if self.original_feature is None:
					dot.node(ctr_node_name, feature_name)
				else:
					dot.node(ctr_node_name, '%s: %s' % (feature_name, self.original_feature[feature_name]))
				
				hash_node = 'interaction_%d_hash_%s' % (ctr_index, feature_name)
				dot.node(hash_node, 'hash_uint64')
				dot.edge(ctr_node_name, hash_node, label='')

				dot.edge(hash_node, interaction_node, label='')

			for feature_name in ctrs_feature['float_features']:
				rule = ctrs_feature['float_featrue_rules'][feature_name]
				
				ctr_node_name = 'interaction_%d_%s' % (ctr_index, feature_name)
				if self.original_feature is None:
					dot.node(ctr_node_name, feature_name)
				else:
					dot.node(ctr_node_name, '%s: %s' % (feature_name, str(self.original_feature[feature_name])))

				rule_node = 'interaction_%d_rele_%s' % (ctr_index, feature_name)
				rule_str = ''
				if rule['type'] == 'compare':
					rule_str = 'if value > %f\n then set 1\nelse set 0' % rule['value']
				elif rule['type'] == 'equal':
					rule_str = 'if value == %f\n then set 1\nelse set 0' % rule['value']
				elif rule['type'] == 'oh_compare':
					rule_str = 'if onehot_map(value) == %f\n then set 1\nelse set 0' % rule['value']
				else: # rule['type'] == 'oh_equal'
					rule_str = 'if onehot_map(value) > %f\n then set 1\nelse set 0' % rule['value']

				dot.node(rule_node, rule_str)
				dot.edge(ctr_node_name, rule_node, label='')

				dot.edge(rule_node, interaction_node, label='')

			generate_actions = interaction_ctr['generate_action']
			for action_index in ctrs_features_dict[ctr_index]:
				action = generate_actions[action_index]
				feature_index = feature_index_dict[ctr_index][action_index]

				ctr_type = action['ctr'].base_ctr_type
				action_node = 'interaction_%d_action_%d' % (ctr_index, action_index)
				dot.node(action_node, ctr_type + ' Type Action')
				dot.edge(interaction_node, action_node, label='action ' + str(action_index))

				disperse_node = 'interaction_%d_disperse_%d' % (ctr_index, feature_index)
				dot.node(disperse_node, 'discrete')
				dot.edge(action_node, disperse_node, label='')

				feature_node = 'feature_%d' % feature_index
				if self.binary_features is None:
					dot.node(feature_node, feature_node)
				else:
					dot.node(feature_node, '%s: %d' % (feature_node, self.binary_features[feature_index]))

				dot.edge(disperse_node, feature_node, label='')

	def _draw_feature_value(self, dot, feature_list, feature_value):
		for index in feature_list:
			featrure_node = 'feature_%d' % index
			value_node = 'feature_value_%d' % index
			dot.node(value_node, str(feature_value[index]))
			dot.edge(featrure_node, value_node, label='value')

	def draw_feature(self, dot, feature_list):
		if feature_list is None:
			return dot
		
		float_features, cat_features, ctrs_features = [], [], []
		for index in feature_list:
			if index < self.float_feature_name_count:
				float_features.append(index)
			elif index < (self.float_feature_name_count + self.one_hot_feature_count):
				cat_features.append(index)
			else:
				ctrs_features.append(index)

		self._draw_float_feature(dot, float_features)
		self._draw_one_hot_feature(dot, cat_features)
		self._draw_ctrs_feature(dot, ctrs_features)

		return dot

	def draw_all_feature(self, dot):

		self._draw_float_feature(dot, range(self.float_feature_name_count))
		self._draw_one_hot_feature(dot, range(self.float_feature_name_count, self.float_feature_name_count + self.one_hot_feature_count))
		self._draw_ctrs_feature(dot, range(self.float_feature_name_count + self.one_hot_feature_count, self.all_feature_count))

		return dot
