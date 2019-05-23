# Import Supervisely format 
This plugin allows you to upload Projects in Supervisely format which includes images, annotations and `meta.json`. More about Supervisely format you can read [here](https://docs.supervise.ly/ann_format/).

For this format the structure of directory should be the following:

```
my_project
├── meta.json
├── dataset_name_01
│   ├── ann
│   │   ├── img_x.json
│   │   ├── img_y.json
│   │   └── img_z.json
│   └── img
│       ├── img_x.jpeg
│       ├── img_y.jpeg
│       └── img_z.jpeg
├── dataset_name_02
│   ├── ann
│   │   ├── img_x.json
│   │   ├── img_y.json
│   │   └── img_z.json
│   └── img
│       ├── img_x.jpeg
│       ├── img_y.jpeg
│       └── img_z.jpeg
```

Directory "my_project" contains two folders and file `meta.json`. For each folder will be created corresponding dataset inside project. As you can see, images are separated from annotations.

### Example
In this example we will upload project with one dataset and will name it "Test Project".

![](https://i.imgur.com/Vuhqur1.gif)
