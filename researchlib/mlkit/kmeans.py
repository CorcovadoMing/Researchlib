import cudf
import cuml
import torch.utils.dlpack
import warnings

def KMeans(data, cluster):
    warnings.filterwarnings('ignore')
    data_pack = torch.utils.dlpack.to_dlpack(data)
    data_df = cudf.from_dlpack(data_pack)
    model = cuml.KMeans(n_clusters=cluster)
    result = model.fit(data_df)
    labels = torch.utils.dlpack.from_dlpack(result.labels_.to_dlpack())
    warnings.filterwarnings('once')
    return labels