# Pedestrian Import

#### Usage steps:
1) Download `Pedestrian` dataset from [official site](http://mmlab.ie.cuhk.edu.hk/projects/luoWTiccv2013DDN/index.html).

   * pedestrian_parsing_dataset.zip	


2) Unpack archive

3) Directory structure have to be the following:

```	
	.	
	└── pedestrian_parsing_dataset	
	    └── data	
	        ├── 1	
	        │   ├── 0024_15.jpg	
	        │   ├── 0024_15_m.png	
	        │   ├── 0024_4.jpg	
	        │   ├── 0024_4_m.png	
	        │   └── ...	
	        ├── 2	
	        │   ├── 0281_2.jpg	
	        │   ├── 0281_2_m.png	
	        │   ├── 0331_1.jpg	
	        │   ├── 0331_1_m.png	
	        │   └── ...	
	        └── ...	
     
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `Pedestrian` import plugin.
5) Select directory (`pedestrian_parsing_dataset`) and drag and drop it to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow datasets: `dataset`.

    ![](https://i.imgur.com/QySzz3z.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/PcHpMfL.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/CnoK6gx.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
