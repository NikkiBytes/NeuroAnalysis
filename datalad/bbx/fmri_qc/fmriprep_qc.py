import glob
import os
import jinja2
import zipfile




def get_data(sub_id, study_path, ses_id, sub_dict):
    # initialize dictionary
    sub_dict[sub_id] = {}

    # Get data:

    # anatomicals:
    t1w_brainmask_svg = os.path.join(study_path,'{}_space-MNI152NLin2009cAsym_T1w.svg'.format(sub_id))
    sub_dict[sub_id]["ANAT"]  = t1w_brainmask_svg

    # fieldmaps:
    fmap_svg = os.path.join(study_path,'{}_ses-{}_desc-brain_mask.svg'.format(sub_id, ses_id))
    sub_dict[sub_id]["FMAP"] = fmap_svg

    # functionals
    func_fmaps=glob.glob(os.path.join(study_path,'{}_ses-{}_*desc-fieldmap_bold.svg'.format(sub_id, ses_id) ))
    func_sdcs=glob.glob(os.path.join(study_path, 'sub-*',  '{}_ses-{}_*desc-sdc_bold.svg'.format(sub_id, ses_id)))
    func_flirts=glob.glob(os.path.join(study_path, 'sub-*', '{}_ses-{}_*_desc-flirtbbr_bold.svg'.format(sub_id, ses_id)))
    sub_dict[sub_id]["FUNC_FMAPS"] = func_fmaps
    sub_dict[sub_id]["FUNC_SDCS"] = func_sdcs
    sub_dict[sub_id]["FUNC_FLIRTS"] = func_flirts

def make_html(sub_dict, base_dir):
    env = jinja2.Environment(loader=jinja2.PackageLoader('app'))
    template = env.get_template('base.html')

    filename = os.path.join(base_dir, "reports/_test.html")

    with open(filename, "w") as fh:
        fh.write(template.render(
            Title = "BBx Session 1 fmriprep QC",
            data_dict = sub_dict

        ))

    print("> Report processing complete, locate generated report here: {}".format(os.path.join(base_dir, filename)))


def main():
    # Get base path, and check for existing QC file.
    #study_path = "/projects/niblab/bids_projects/Experiments/bbx"
    base_dir=os.path.dirname(os.path.abspath(__file__))
    print("> FOUND THE DIRECTORY PATH: {} \nStarting program.....".format(base_dir))
    zip_imgs_path = os.path.join(base_dir, "fmriprep_images.zip")
    imgs_path = os.path.join(base_dir,"fmriprep_images")
    sess_id = "ses-1"
    with zipfile.ZipFile(zip_imgs_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
    # get fmriprep path of all subjects --here we have set it to ses-2 (**may have to customize)
    sub_ids = [x.split("/")[-1].split("_")[0] for x in glob.glob(os.path.join(imgs_path, 'sub-00*'))]
    sub_ids = list(set(sub_ids))
    sub_dict = {}
    #print(sub_ids)
    for sub_id in sorted(sub_ids):
        get_data(sub_id, imgs_path, "1", sub_dict)
    print("> FOUND DATA SUBJECTS {} MAKING HTML REPORT....".format(sorted(sub_dict.keys())))
    #for i in sub_dict:
     #   print(sub_dict[i]["FMAP"])
    make_html(sub_dict, base_dir)

if __name__ == "__main__":
    main()