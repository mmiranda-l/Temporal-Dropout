import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import OneHotEncoder
from src.datasets.utils import storage_set, reverse_padding

INPUT_FEATURES = {
    "S2": ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
	"Evapotranspiration": ['ETref', 'ETc', 'ETcadj', 'Kc'],
    "weather": ['day_rain', 'cum_rain', 'avg_temp', 'max_temp', 'min_temp'], 
	"S2SCL": ['scl_class'], 
	"DAS": ["das"],
}

#das: number of days after sowing (perhaps in temporal models )

def extract_data(df_input, df_yield, padd="pre"):
	list_df = df_input.groupby(level=0).agg(list)
	index_yield_align = list(df_yield.index)

	view_data = {}
	for feats in INPUT_FEATURES:
		print(f" Processing : {feats}")
		feat_data = pd.concat([list_df[v].apply(pd.Series).rename(columns={i: f"{v}({i})" for i in range(100)}) for v in INPUT_FEATURES[feats]], axis=1, join="inner").loc[index_yield_align].values

		if "S2SCL" == feats:
			if padd == "pre":
				feat_data = np.apply_along_axis(reverse_padding, 1, feat_data)
			
			enc = OneHotEncoder()
			data = enc.fit_transform(feat_data.flatten().reshape(-1,1)).toarray()
			print(f"Fitted with categories= {enc.categories_}")
			view_data[feats] = data.reshape(len(feat_data), -1, len(enc.categories_[0]))
			if padd =="post":
				mask_indx = np.isnan(feat_data)
				view_data[feats][mask_indx] = np.nan
				view_data[feats] = view_data[feats].astype(np.float32) 
			else:
				view_data[feats] = view_data[feats].astype(np.uint8) 
		else:
			feat_data = feat_data.reshape(len(feat_data), len(INPUT_FEATURES[feats]), -1).transpose([0,2,1])
			if padd == "pre":
				feat_data = np.apply_along_axis(reverse_padding, 1, feat_data) #, pad_val=0)
			view_data[feats] = feat_data.astype(np.float32) 
	
	print("features created are :" , list(view_data.keys()))
	return index_yield_align, view_data, df_yield["dry_yield"].values.astype(np.float32)

if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
	    "--data_dir",
	    "-d",
	    required=True,
	    type=str,
	    help="path of the data directory",
	)
	arg_parser.add_argument(
	    "--out_dir",
	    "-o",
	    required=True,
	    type=str,
	    help="path of the output directory to store the data",
	)
	arg_parser.add_argument(
	    "--crop",
	    "-c",
	    required=True,
	    type=str,
	    help="type of crop to be used, options are [cereals, winterwheat]",
	)
	arg_parser.add_argument(
	    "--padd",
	    "-p",
	    type=str,
	    help="type of padding, options are [pre, post]",
		default="pre"
	)
	args = arg_parser.parse_args()

	if "cereals" in args.crop.lower():	
		df_input = pd.read_csv(f"{args.data_dir}/Cereals_covariates_tot_evp.csv")
		df_yield = pd.read_csv(f"{args.data_dir}/Cereals_yield_tot.csv")
	elif "wheat" in args.crop.lower():	
		df_input = pd.read_csv(f"{args.data_dir}/WW_covariates_tot.csv")
		df_yield = pd.read_csv(f"{args.data_dir}/WW_yield_tot.csv")
	else:
		raise Exception("Unrecognized crop")

	df_input["identifier"] = df_input["coord_id"] + "_"+ df_input["harvest_year"].astype(str)
	df_input.set_index("identifier",inplace=True)
	df_yield["identifier"] = df_yield["coord_id"] + "_"+ df_yield["harvest_year"].astype(str)
	df_yield.set_index("identifier",inplace=True)

	print(f"CREATING AND SAVING DATA WITH CROP AS = {args.crop}")

	indx_data, views_data, target_data = extract_data(df_input, df_yield, padd=args.padd.lower())
	print(f"Dataset with {len(views_data)} views, and and shape",{v: views_data[v].shape for v in views_data})
	
	views_data["coords"] = df_yield[["x_coord","y_coord"]].values.astype(np.uint16)
	views_data["year"] = df_yield["harvest_year"].values.astype(np.int16)
	views_data["field_id"] = df_yield["FID"].values
	
	storage_set([indx_data, views_data, target_data], args.out_dir, name=f"yield_{args.crop}", mode="")