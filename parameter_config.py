
MNIST_STD_MODEL = "/kaggle/input/models/niccolosici/mnistpt-std-gelu/other/default/1/standard_model_gelu.pth"#"/kaggle/working/models/standard_model_relu.pth"#"/kaggle/working/models/standard_model_relu.pth"
MNIST_WIDE_MODEL = "/kaggle/input/models/niccolosici/mnistpt-wide-gelu/other/default/1/wide_model_gelu.pth"#"/kaggle/working/models/wide_model_relu.pth"
FMNIST_STD_MODEL = "/kaggle/input/models/niccolosici/fashionmnistft-std-gelu/other/default/1/standard_finetuned_gelu.pth"#/kaggle/working/models/standard_finetuned_relu.pth"
FMNIST_WIDE_MODEL = "/kaggle/input/models/niccolosici/fashionmnistft-wide-gelu/other/default/1/wide_finetuned_gelu.pth"#"/kaggle/working/models/wide_finetuned_relu.pth"

TRAIN = True
gvendi_params = {                           
  'dataset': datasets.FashionMNIST,       
  'mean': (0.2860,),                  
  'std': (0.3530,),                                                                                                                                                                                                                                
  'proxy_weights': MNIST_STD_MODEL,                                                                                                                                                                                                                
  'proxy_hidden_size_1': 256,                                                                                                                                                                                                                      
  'proxy_hidden_size_2': 128,                                                                                                                                                                                                                      
  'subset_size': 20,                                                                                                                                                                                                                            
  'n_subsets': 10,                                                                                                                                                                                                                                  
  'n_pool': 5000,                                                                                                                                                                                                                                
  'k_per_class': 10,                                                                                                                                                                                                                               
  'proj_dim': 1024,                                                                                                                                                                                                                                
  'n_classes': 10,                        
}                                                                                                                                                                                                                                                    
'''
'to_extract': [                                                                                                                                                                                                                                  
      {'using': 'standard', 'method': method_gvendi, 'params': gvendi_params, 'filename': 'standard_fashion'},                                                                                                                                     
      {'using': 'wide',     'method': method_gvendi, 'params': gvendi_params, 'filename': 'wide_fashion'},                                                                                                                                         
      {                                                                                                                                                                                                                                                    
      'using': 'wide',                                                                                                                                                                                                                                 
      'method': method_full,                                                                                                                                                                                                                           
      'params': {'dataset': datasets.FashionMNIST, 'mean': (0.2860,), 'std': (0.3530,)},                                                                                                                                                               
      'filename': 'wide_fashion_activations', 
      },
      {                                                                                                                                                                                                                                                    
      'using': 'standard',                                                                                                                                                                                                                                 
      'method': method_full,                                                                                                                                                                                                                           
      'params': {'dataset': datasets.FashionMNIST, 'mean': (0.2860,), 'std': (0.3530,)},                                                                                                                                                               
      'filename': 'standard_fashion_activations', 
      },
  ], 
'''  

EXTRACT_ACTIVATION = {                                                                                                                                                                                                                               
  'models': {
      'standard': {'hidden_size_1': 256,  'hidden_size_2': 128, 'weights': MNIST_STD_MODEL},                                                                                                                                                       
      'standard_ft': {'hidden_size_1': 256,  'hidden_size_2': 128, 'weights': FMNIST_STD_MODEL},                                                                                                                                                       
      'wide':     {'hidden_size_1': 1024, 'hidden_size_2': 512, 'weights': MNIST_WIDE_MODEL},           
      'wide_ft': {'hidden_size_1': 1024, 'hidden_size_2': 512, 'weights': FMNIST_WIDE_MODEL},           
  },                                      
  'N': 10000,                                                                                                                                                                                                                                      
  'save_dir': '/kaggle/working',                                                                                                                                                                                                                   
  'to_extract': [ 
      {'using': 'standard',    'method': method_gvendi_interpolated, 'params': {**gvendi_params, 'interp_number': 100, 'seed': 42}, 'filename': 'standard_fashion'},
      {'using': 'wide',        'method': method_gvendi_interpolated, 'params': {**gvendi_params, 'interp_number': 100, 'seed': 42}, 'filename': 'wide_fashion'},
  
      #{'using': 'standard', 'method': method_gvendi, 'params': gvendi_params, 'filename': 'standard_fashion'},                                                                                                                                     
      #{'using': 'wide',     'method': method_gvendi, 'params': gvendi_params, 'filename': 'wide_fashion'},            
      
      #{'using': 'wide_ft',     'method': method_gvendi, 'params': gvendi_params, 'filename': 'wide_ft_fashion'},       
      #{'using': 'standard_ft', 'method': method_gvendi, 'params': gvendi_params, 'filename': 'standard_ft_fashion'},   
     
      
  ],
}
'''
'tools': [                                                                                                                                                                                                                              
          #{'method': tool_cov_spectrum_dist, 'params': {'normalize': True}, 'name': 'Cov spectrum dist'},
          #{'method': tool_sv_sum,            'params': {'normalize': True}, 'name': 'Sum of singular values'},         
          {'method': tool_norm_analysis, 'params': {
              'full_source': 'wide_fashion_activations',
              'full_target': 'standard_fashion_activations',
            }, 'name': 'Norm analysis'},    
      ],            

'preprocessing': {
          'method': None,#preprocess_mean_center_frobenius,
          'params': {},                   
      },
'''
COMPUTE_ORTHO_MAP = {                                                                                                                                                                                                                                
      'preprocessing': {
          'method': None,#preprocess_mean_center_frobenius,
          'params': {},                   
      },
                                                                                                                                                                                                        
      'save_dir': '/kaggle/working',                                                                                                                                                                                                                 
      'act_dir': '/kaggle/working/activations',                                                                                                                                                                                                        
      'to_compute': [
          {'source': 'wide_fashion_gvendi17.1_sub0',   'target': 'standard_fashion_gvendi17.1_sub0',   'name': 'wide_to_std_gvendi_fashion_17.1_sub0',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi16.6_sub1',   'target': 'standard_fashion_gvendi16.6_sub1',   'name': 'wide_to_std_gvendi_fashion_16.6_sub1',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi16.9_sub2',   'target': 'standard_fashion_gvendi16.9_sub2',   'name': 'wide_to_std_gvendi_fashion_16.9_sub2',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi16.5_sub3',   'target': 'standard_fashion_gvendi16.5_sub3',   'name': 'wide_to_std_gvendi_fashion_16.5_sub3',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi16.9_sub4',   'target': 'standard_fashion_gvendi16.9_sub4',   'name': 'wide_to_std_gvendi_fashion_16.9_sub4',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.6_sub5',   'target': 'standard_fashion_gvendi15.6_sub5',   'name': 'wide_to_std_gvendi_fashion_15.6_sub5',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.3_sub6',   'target': 'standard_fashion_gvendi15.3_sub6',   'name': 'wide_to_std_gvendi_fashion_15.3_sub6',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.6_sub7',   'target': 'standard_fashion_gvendi15.6_sub7',   'name': 'wide_to_std_gvendi_fashion_15.6_sub7',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi18.1_sub8',   'target': 'standard_fashion_gvendi18.1_sub8',   'name': 'wide_to_std_gvendi_fashion_18.1_sub8',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi14.5_sub9',   'target': 'standard_fashion_gvendi14.5_sub9',   'name': 'wide_to_std_gvendi_fashion_14.5_sub9',   'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi16.8_sub10',  'target': 'standard_fashion_gvendi16.8_sub10',  'name': 'wide_to_std_gvendi_fashion_16.8_sub10',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.5_sub11',  'target': 'standard_fashion_gvendi15.5_sub11',  'name': 'wide_to_std_gvendi_fashion_15.5_sub11',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.2_sub12',  'target': 'standard_fashion_gvendi15.2_sub12',  'name': 'wide_to_std_gvendi_fashion_15.2_sub12',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.9_sub13',  'target': 'standard_fashion_gvendi15.9_sub13',  'name': 'wide_to_std_gvendi_fashion_15.9_sub13',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi14.2_sub14',  'target': 'standard_fashion_gvendi14.2_sub14',  'name': 'wide_to_std_gvendi_fashion_14.2_sub14',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi16.1_sub15',  'target': 'standard_fashion_gvendi16.1_sub15',  'name': 'wide_to_std_gvendi_fashion_16.1_sub15',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi17.3_sub16',  'target': 'standard_fashion_gvendi17.3_sub16',  'name': 'wide_to_std_gvendi_fashion_17.3_sub16',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.3_sub17',  'target': 'standard_fashion_gvendi15.3_sub17',  'name': 'wide_to_std_gvendi_fashion_15.3_sub17',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi16.9_sub18',  'target': 'standard_fashion_gvendi16.9_sub18',  'name': 'wide_to_std_gvendi_fashion_16.9_sub18',  'N': 20, 'ranks': [150, 150, 150, 150]},
          {'source': 'wide_fashion_gvendi15.7_sub19',  'target': 'standard_fashion_gvendi15.7_sub19',  'name': 'wide_to_std_gvendi_fashion_15.7_sub19',  'N': 20, 'ranks': [150, 150, 150, 150]},
      ],                                                                                                                                                                                                                                               
  }   
_SUBSETS = [
    ('gvendi86.1_sub0',  500, 32.10), ('gvendi83.8_sub1',  500, 23.54), ('gvendi85.9_sub2',  500, 28.48),
    ('gvendi83.4_sub3',  500, 30.89), ('gvendi72.6_sub4',  500, 33.91), ('gvendi76.2_sub5',  500, 23.83),
    ('gvendi77.5_sub6',  500, 23.95), ('gvendi76.3_sub7',  500, 31.51), ('gvendi80.5_sub8',  500, 31.40),
    ('gvendi92.1_sub9',  500, 35.20), ('gvendi86.8_sub10', 500, 28.20), ('gvendi80.9_sub11', 500, 26.28),
    ('gvendi84.5_sub12', 500, 23.38), ('gvendi79.1_sub13', 500, 23.50), ('gvendi88.3_sub14', 500, 22.34),
    ('gvendi82.4_sub15', 500, 25.62), ('gvendi91.3_sub16', 500, 23.12), ('gvendi82.6_sub17', 500, 30.36),
    ('gvendi77.6_sub18', 500, 28.16), ('gvendi82.5_sub19', 500, 22.47),
]
_SUBSETS_20 = [
    ('gvendi17.1_sub0',  20, 0.0), ('gvendi16.6_sub1',  20, 0.0), ('gvendi16.9_sub2',  20, 0.0),
    ('gvendi16.5_sub3',  20, 0.0), ('gvendi16.9_sub4',  20, 0.0), ('gvendi15.6_sub5',  20, 0.0),
    ('gvendi15.3_sub6',  20, 0.0), ('gvendi15.6_sub7',  20, 0.0), ('gvendi18.1_sub8',  20, 0.0),
    ('gvendi14.5_sub9',  20, 0.0)
]
'''
('gvendi17.1_sub0',  20, 0.0), ('gvendi16.6_sub1',  20, 0.0), ('gvendi16.9_sub2',  20, 0.0),
    ('gvendi16.5_sub3',  20, 0.0), ('gvendi16.9_sub4',  20, 0.0), ('gvendi15.6_sub5',  20, 0.0),
    ('gvendi15.3_sub6',  20, 0.0), ('gvendi15.6_sub7',  20, 0.0), ('gvendi18.1_sub8',  20, 0.0),
    ('gvendi14.5_sub9',  20, 0.0) ('gvendi16.8_sub10', 20, 0.0), ('gvendi15.5_sub11', 20, 0.0),
    ('gvendi15.2_sub12', 20, 0.0), ('gvendi15.9_sub13', 20, 0.0), ('gvendi14.2_sub14', 20, 0.0),
    ('gvendi16.1_sub15', 20, 0.0), ('gvendi17.3_sub16', 20, 0.0), ('gvendi15.3_sub17', 20, 0.0),
    ('gvendi16.9_sub18', 20, 0.0), ('gvendi15.7_sub19', 20, 0.0),
'''

ANALYSIS_CONFIG = {
    'analyses': [
        {
            'load_method': load_paired_boundary_data,
            'load_params': {
                'act_dir': '/kaggle/working/activations',
                'suffix': '_relu',
                'wide_pre': f'wide_fashion_{tag}',
                'wide_ft': f'wide_ft_fashion_{tag}',
                'std_pre': f'standard_fashion_{tag}',
                'std_ft': f'standard_ft_fashion_{tag}',
            },
            'experiment': paired_boundary_containment,
            'experiment_param': {'k': 10, 'boundary_min_frac': 0.4},
            'acc': acc,
        }
        for tag, N, acc in _SUBSETS_20
    ]
}

CCTA_ANALYSIS = {                                                                                                                                                                                                                                    
    'preprocessing': {                                                                                                                                                                                                                                 
        'method': preprocess_mean_center_frobenius,                                                                                                                                                                                                    
        'params': {},                                                                                                                                                                                                                                  
    },                                                                                                                                                                                                                                                 
    'save_dir': '/kaggle/working',                                                                                                                                                                                                                     
    'act_dir': '/kaggle/working/activations',                                                                                                                                                                                                          
    'subsets': [                                                                                                                                                                                                                                       
        {'source': 'wide_fashion_gvendi92.1_sub9',   'target': 'standard_fashion_gvendi92.1_sub9',   'name': '92.1_sub9',   'N': 500, 'alpha': 3.5},   # 35.20                                                                                         
        {'source': 'wide_fashion_gvendi72.6_sub4',   'target': 'standard_fashion_gvendi72.6_sub4',   'name': '72.6_sub4',   'N': 500, 'alpha': 4.8},   # 33.91                                                                                         
        {'source': 'wide_fashion_gvendi86.1_sub0',   'target': 'standard_fashion_gvendi86.1_sub0',   'name': '86.1_sub0',   'N': 500, 'alpha': 2.5},   # 32.10                                                                                         
        {'source': 'wide_fashion_gvendi76.3_sub7',   'target': 'standard_fashion_gvendi76.3_sub7',   'name': '76.3_sub7',   'N': 500, 'alpha': 4.5},   # 31.51                                                                                         
        {'source': 'wide_fashion_gvendi80.5_sub8',   'target': 'standard_fashion_gvendi80.5_sub8',   'name': '80.5_sub8',   'N': 500, 'alpha': 5.8},   # 31.40                                                                                         
        {'source': 'wide_fashion_gvendi83.4_sub3',   'target': 'standard_fashion_gvendi83.4_sub3',   'name': '83.4_sub3',   'N': 500, 'alpha': 3.4},   # 30.89                                                                                         
        {'source': 'wide_fashion_gvendi82.6_sub17',  'target': 'standard_fashion_gvendi82.6_sub17',  'name': '82.6_sub17',  'N': 500},                 # 30.36                                                                                         
        {'source': 'wide_fashion_gvendi85.9_sub2',   'target': 'standard_fashion_gvendi85.9_sub2',   'name': '85.9_sub2',   'N': 500, 'alpha': 2.2},   # 28.48                                                                                         
        {'source': 'wide_fashion_gvendi86.8_sub10',  'target': 'standard_fashion_gvendi86.8_sub10',  'name': '86.8_sub10',  'N': 500},                 # 28.20                                                                                         
        {'source': 'wide_fashion_gvendi77.6_sub18',  'target': 'standard_fashion_gvendi77.6_sub18',  'name': '77.6_sub18',  'N': 500},                 # 28.16                                                                                         
        {'source': 'wide_fashion_gvendi80.9_sub11',  'target': 'standard_fashion_gvendi80.9_sub11',  'name': '80.9_sub11',  'N': 500},                 # 26.28                                                                                         
        {'source': 'wide_fashion_gvendi82.4_sub15',  'target': 'standard_fashion_gvendi82.4_sub15',  'name': '82.4_sub15',  'N': 500},                 # 25.62                                                                                         
        {'source': 'wide_fashion_gvendi77.5_sub6',   'target': 'standard_fashion_gvendi77.5_sub6',   'name': '77.5_sub6',   'N': 500, 'alpha': 5.0},   # 23.95                                                                                         
        {'source': 'wide_fashion_gvendi76.2_sub5',   'target': 'standard_fashion_gvendi76.2_sub5',   'name': '76.2_sub5',   'N': 500, 'alpha': 3.2},   # 23.83                                                                                         
        {'source': 'wide_fashion_gvendi83.8_sub1',   'target': 'standard_fashion_gvendi83.8_sub1',   'name': '83.8_sub1',   'N': 500, 'alpha': 2.2},   # 23.54                                                                                         
        {'source': 'wide_fashion_gvendi79.1_sub13',  'target': 'standard_fashion_gvendi79.1_sub13',  'name': '79.1_sub13',  'N': 500},                 # 23.50                                                                                         
        {'source': 'wide_fashion_gvendi84.5_sub12',  'target': 'standard_fashion_gvendi84.5_sub12',  'name': '84.5_sub12',  'N': 500},                 # 23.38                                                                                         
        {'source': 'wide_fashion_gvendi91.3_sub16',  'target': 'standard_fashion_gvendi91.3_sub16',  'name': '91.3_sub16',  'N': 500},                 # 23.12                                                                                         
        {'source': 'wide_fashion_gvendi82.5_sub19',  'target': 'standard_fashion_gvendi82.5_sub19',  'name': '82.5_sub19',  'N': 500},                 # 22.47                                                                                         
        {'source': 'wide_fashion_gvendi88.3_sub14',  'target': 'standard_fashion_gvendi88.3_sub14',  'name': '88.3_sub14',  'N': 500},                 # 22.34                                                                                         
    ],                                                                                                                                                                                                                                                 
  }
         

GEOMETRIC_ANALYSIS = {
    'act_dir': '/kaggle/working/activations',
    'k': 10,
    'subsets': [
        {'source': 'wide_fashion_gvendi92.1_sub9',   'target': 'standard_fashion_gvendi92.1_sub9',   'name': '92.1_sub9',   'N': 500},
        {'source': 'wide_fashion_gvendi72.6_sub4',   'target': 'standard_fashion_gvendi72.6_sub4',   'name': '72.6_sub4',   'N': 500},
        {'source': 'wide_fashion_gvendi86.1_sub0',   'target': 'standard_fashion_gvendi86.1_sub0',   'name': '86.1_sub0',   'N': 500},
        {'source': 'wide_fashion_gvendi76.3_sub7',   'target': 'standard_fashion_gvendi76.3_sub7',   'name': '76.3_sub7',   'N': 500},
        {'source': 'wide_fashion_gvendi80.5_sub8',   'target': 'standard_fashion_gvendi80.5_sub8',   'name': '80.5_sub8',   'N': 500},
        {'source': 'wide_fashion_gvendi83.4_sub3',   'target': 'standard_fashion_gvendi83.4_sub3',   'name': '83.4_sub3',   'N': 500},
        {'source': 'wide_fashion_gvendi82.6_sub17',  'target': 'standard_fashion_gvendi82.6_sub17',  'name': '82.6_sub17',  'N': 500},
        {'source': 'wide_fashion_gvendi85.9_sub2',   'target': 'standard_fashion_gvendi85.9_sub2',   'name': '85.9_sub2',   'N': 500},
        {'source': 'wide_fashion_gvendi86.8_sub10',  'target': 'standard_fashion_gvendi86.8_sub10',  'name': '86.8_sub10',  'N': 500},
        {'source': 'wide_fashion_gvendi77.6_sub18',  'target': 'standard_fashion_gvendi77.6_sub18',  'name': '77.6_sub18',  'N': 500},
        {'source': 'wide_fashion_gvendi80.9_sub11',  'target': 'standard_fashion_gvendi80.9_sub11',  'name': '80.9_sub11',  'N': 500},
        {'source': 'wide_fashion_gvendi82.4_sub15',  'target': 'standard_fashion_gvendi82.4_sub15',  'name': '82.4_sub15',  'N': 500},
        {'source': 'wide_fashion_gvendi77.5_sub6',   'target': 'standard_fashion_gvendi77.5_sub6',   'name': '77.5_sub6',   'N': 500},
        {'source': 'wide_fashion_gvendi76.2_sub5',   'target': 'standard_fashion_gvendi76.2_sub5',   'name': '76.2_sub5',   'N': 500},
        {'source': 'wide_fashion_gvendi83.8_sub1',   'target': 'standard_fashion_gvendi83.8_sub1',   'name': '83.8_sub1',   'N': 500},
        {'source': 'wide_fashion_gvendi79.1_sub13',  'target': 'standard_fashion_gvendi79.1_sub13',  'name': '79.1_sub13',  'N': 500},
        {'source': 'wide_fashion_gvendi84.5_sub12',  'target': 'standard_fashion_gvendi84.5_sub12',  'name': '84.5_sub12',  'N': 500},
        {'source': 'wide_fashion_gvendi91.3_sub16',  'target': 'standard_fashion_gvendi91.3_sub16',  'name': '91.3_sub16',  'N': 500},
        {'source': 'wide_fashion_gvendi82.5_sub19',  'target': 'standard_fashion_gvendi82.5_sub19',  'name': '82.5_sub19',  'N': 500},
        {'source': 'wide_fashion_gvendi88.3_sub14',  'target': 'standard_fashion_gvendi88.3_sub14',  'name': '88.3_sub14',  'N': 500},
    ],
}

    #PARAMETER-CONFIG



USE_RELU = True
TRAIN_CONFIG = {                                                                                                                                                                                                                                     
      'target_model': {                                                                                                                                                                                                                                
          'name': 'standard',                                                                                                                                                                                                                          
          'hidden_size_1': 256,                                                                                                                                                                                                                        
          'hidden_size_2': 128,
      },                                                                                                                                                                                                                                               
      'source_model': {                                                                                                                                                                                                                              
          'name': 'wide',                                                                                                                                                                                                                              
          'hidden_size_1': 1024,                                                                                                                                                                                                                     
          'hidden_size_2': 512,               
      },                                  
      'pretrained_dataset': 'mnist',
      'finetuned_dataset': 'fashion',                                                                                                                                                                                                                  
      'pretrain_epochs': 5,
      'pretrain_lr': 1e-3,                                                                                                                                                                                                                             
      'finetune_epochs': 7,                                                                                                                                                                                                                          
      'finetune_lr': 1e-4,                
      'batch_size_train': 128,
      'batch_size_test': 256,                                                                                                                                                                                                                          
      'save_dir': '/kaggle/working/models',
  }  

COMPUTE_FMAPS = [
      (f'wide_fashion_{tag}', f'standard_fashion_{tag}', f'fmap_wide_to_std_{tag}',15 )
      for tag, N, acc in _SUBSETS_20
  ]+[
      (f'wide_fashion_{tag}', f'standard_fashion_{tag}', f'fmap_wide_to_std_{tag}',14 )
      for tag, N, acc in _SUBSETS_20
  ]+[
      (f'wide_fashion_{tag}', f'standard_fashion_{tag}', f'fmap_wide_to_std_{tag}',13 )
      for tag, N, acc in _SUBSETS_20
  ]
                                                                                                                                                                                                                                        
TRANSPORT_ORTHO =[
    (f'procrustes_matrices_procrustes_matrices_fmap_wide_to_std_{tag}'+'_15', f'fmap_ortho_{tag}_15')
    for tag, N, acc in _SUBSETS_20
]+[
    (f'procrustes_matrices_procrustes_matrices_fmap_wide_to_std_{tag}'+'_14', f'fmap_ortho_{tag}_14')
    for tag, N, acc in _SUBSETS_20
]+[
    (f'procrustes_matrices_procrustes_matrices_fmap_wide_to_std_{tag}'+'_13', f'fmap_ortho_{tag}_13')
    for tag, N, acc in _SUBSETS_20
]

CENTROID_ANALYSIS=[
    ("standard_fashion_gvendi86.1_sub0_activations", "fashion", "standard"),
    ("standard_fashion_gvendi85.9_sub2_activations", "fashion", "standard"),
    ("standard_fashion_gvendi83.8_sub1_activations", "fashion", "standard"),
    ("standard_fashion_gvendi83.4_sub3_activations", "fashion", "standard"),
    ("standard_fashion_gvendi72.6_sub4_activations", "fashion", "standard")
    
]
PROCRUSTES_RESIDUAL = [
    ('procrustes_matrices_wide_to_std_gvendi_fashion_86.1_500', 'wide_fashion_activations', 'standard_fashion_activations', "fashion",'gvendi86.1'),
    ('procrustes_matrices_wide_to_std_gvendi_fashion_85.9_500', 'wide_fashion_activations', 'standard_fashion_activations', "fashion",'gvendi85.9'),
    ('procrustes_matrices_wide_to_std_gvendi_fashion_83.8_500', 'wide_fashion_activations', 'standard_fashion_activations', "fashion",'gvendi83.8'),
    ('procrustes_matrices_wide_to_std_gvendi_fashion_83.4_500', 'wide_fashion_activations', 'standard_fashion_activations', "fashion",'gvendi83.4'),
    ('procrustes_matrices_wide_to_std_gvendi_fashion_72.6_500', 'wide_fashion_activations', 'standard_fashion_activations', "fashion",'gvendi72.6')
]                           

VISUALIZE_FMAP_CORR = [
    'fmap_dict_fmap_wide_to_std_interp_fmnist_anchor',
]






EXTRACT_FMAP_ORTHO = [
    (f'fmap_dict_fmap_wide_to_std_{tag}'+'_15', f'procrustes_matrices_fmap_wide_to_std_{tag}'+'_15')
    for tag, N, acc in _SUBSETS_20
]+ [
    (f'fmap_dict_fmap_wide_to_std_{tag}'+'_14', f'procrustes_matrices_fmap_wide_to_std_{tag}'+'_14')
    for tag, N, acc in _SUBSETS_20
]+ [
    (f'fmap_dict_fmap_wide_to_std_{tag}'+'_13', f'procrustes_matrices_fmap_wide_to_std_{tag}'+'_13')
    for tag, N, acc in _SUBSETS_20
]



TRANSPORT_LINEAR = [
    ('linear_matrices_wide_to_std_fashion_regs500.5-500.5-500.5-500.5', "different_reg")
]
COMPUTE_LINEAR_MAP = [
      ('wide_fashion_activations', 'standard_fashion_activations', 'wide_to_std_fashion', [500.5, 500.5, 500.5, 500.5], [None, None, None, None]),                                                                                                                             
  ]  
VISUALIZE_ACTIVATIONS = [                                                                                                                                                                                                                            
  ('wide_mnist_activations', "procrustes_matrices_wide_to_std_fashion",'wide_to_std_fmnist'),
  ('wide_mnist_activations', "procrustes_matrices_wide_to_std_random",'wide_to_std_random'),
]    
PRINT_RANK = [
    'wide_fashion_activations',
    'standard_fashion_activations',
    'wide_fashion_interp_activations',
    'standard_fashion_interp_activations',
]