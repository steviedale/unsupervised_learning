The repository containing all of my code can be found at https://github.com/steviedale/unsupervised_learning

To run the experiments for this project please run the following scripts:

Step 0:
    - Generate dataset info:
        - run step_0_dataset_info.ipynb

Step 1:
	- To generate experimental data:
        - step_1_KMeans.ipynb
        - step_1_GMM.ipynb
    
	- To create plots of the data:
        - run step_1_KMeans_mnist_plot_cluster.ipynb
        - run step_1_GMM_mnist_plot_cluster.ipynb
        - run step_1_KMeans_wine_quality_plot_cluster.ipynb
        - run step_1_GMM_wine_quality_plot_cluster.ipynb
        - run step_1_plot_metrics.ipynb
        - run step_1_plot_metrics_all_datasets_kmeans.ipynb
        - run step_1_plot_metrics_all_datasets_gmm.ipynb

    - To create table:
        - run step_1_get_tables.ipynb

Step 2:
	- To generate experimental data:
		- run step_2_ICA.ipynb
		- run step_2_LLE.ipynb
		- run step_2_PCA.ipynb
		- run step_2_RP.ipynb

	- To create plots of the data: 	
        - run step_2_mnist_DR_2D_plot.ipynb
        - run step_2_wine_quality_DR_2D_plot.ipynb
		- run step_2_plot_metrics.ipynb
        - run step_2_PCA_exp_var_plot.ipynb

    - To create table:
        - run step_2_get_tables.ipynb

Step 3:
	- To generate experimental data:
		- run step_3_ICA_transform_data.ipynb
		- run step_3_LLE_transform_data.ipynb
		- run step_3_PCA_transform_data.ipynb
        - run step_3_RP_transform_data.ipynb
		- run step_3_GMM.ipynb
		- run step_3_KMeans.ipynb

	- To create plots of the data:
		- run step_3_plot_metrics_adjusted_mutual_info.ipynb
		- run step_3_plot_metrics.ipynb
        - run step_3_KMeans_mnist_plot_clusters.ipynb
        - run step_3_plot_metrics_ICA_vs_PCA.ipynb
        - run step_3_GMM_mnist_plot_clusters.ipynb 
        - run step_3_GMM_wine_quality_plot_clusters.ipynb 
        
Step 4:
	- To generate experimental data:
		- run step_4_baseline.ipynb
		- run step_4_PCA.ipynb
		- run step_4_ICA.ipynb
		- run step_4_LLE.ipynb
		- run step_4_RP.ipynb
	- To create plots of the data:
		- run step_4_plot_metrics.ipynb

Step 5:
	- To generate experimental data:
		- run step_5_KMeans_transform_data.ipynb
		- run step_5_train_models.ipynb
	- To create plots of the data:	
		- run step_5_plot_metrics.ipynb

Experimental data will be saved to the 'results' folder and plots will be saved to the 'figures' folder.