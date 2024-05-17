import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

patch_size = 4  # 一个patch有4个像素点
d_model = 32  # Embedding Size
model_dim_C = 32  # 一开始的patch embedding的大小

window_size = 4  # 带窗transformer的窗大小
num_head = 2

merge_size = 2  # 2 * 2的patch组合成1个patch


# 难点1 patch embedding
def image2emb_naive(image, patch_size, weight):
    """ 直观方法去实现patch embedding """
    # image shape: bs*channel*h*w
    patch = F.unfold(image, kernel_size=(patch_size, patch_size),
                     stride=(patch_size, patch_size)).transpose(-1, -2) #[bs, num_patch, patch_depth]
    patch_embedding = patch @ weight #[bs,num_patch,model_dim_C]
    return patch_embedding

def image2emb_conv(image, kernel, stride):
    """ 基于二维卷积来实现patch embedding，embedding的维度就是卷积的输出通道数 """
    conv_output = F.conv2d(image, kernel, stride=stride) # bs*oc*oh*ow
    bs, oc, oh, ow = conv_output.shape
    patch_embedding = conv_output.reshape((bs, oc, oh*ow)).transpose(-1, -2) #[bs,num_patch,model_dim_C]
    return patch_embedding


# ## 2、如何构建MHSA并计算其复杂度？
# - 基于输入x进行三个映射分别得到q,k,v
#     - 此步复杂度为 $3LC^2$，其中L为序列长度，C为特征大小
# - 将q,k,v拆分成多头的形式，注意这里的多头各自计算不影响，所以可以与bs维度进行统一看待
# - 计算$q k^T$，并考虑可能的掩码，即让无效的两两位置之间的能量为负无穷，掩码是在shift window MHSA中会需要，而在window MHSA中暂不需要
#     - 此步复杂度为$L^2C$
# - 计算概率值与v的乘积
#     - 此步复杂度为$L^2C$
# - 对输出进行再次映射
#     - 此步复杂度为$LC^2$
# - 总体复杂度为$4LC^2+2L^2C$

# In[2]:


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, model_dim, num_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        
        self.proj_linear_layer = nn.Linear(model_dim, 3*model_dim)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self, input, additive_mask=None):
        bs, seqlen, model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim // num_head

        proj_output = self.proj_linear_layer(input) # [bs, seqlen, 3*model_dim] 
        
        q, k, v = proj_output.chunk(3, dim=-1) # 3*[bs, seqlen, model_dim] 

        q = q.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2) # [bs, num_head, seqlen, head_dim]
        q = q.reshape(bs*num_head, seqlen, head_dim)

        k = k.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2) # [bs, num_head, seqlen, head_dim]
        k = k.reshape(bs*num_head, seqlen, head_dim)

        v = v.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2) # [bs, num_head, seqlen, head_dim]
        v = v.reshape(bs*num_head, seqlen, head_dim)

        #if additive_mask is None:
        attn_prob = F.softmax(torch.bmm(q, k.transpose(-2, -1))/math.sqrt(head_dim), dim=-1)
        #else:
            #additive_mask = additive_mask.tile((num_head, 1, 1))
            #attn_prob = F.softmax(torch.bmm(q, k.transpose(-2, -1))/math.sqrt(head_dim)+additive_mask, dim=-1)

        output = torch.bmm(attn_prob, v) # [bs*num_head, seqlen, head_dim]
        output = output.reshape(bs, num_head, seqlen, head_dim).transpose(1, 2) #[bs, seqlen, num_head, head_dim]
        output = output.reshape(bs, seqlen, model_dim)
        output = self.final_linear_layer(output)
        return attn_prob, output
        


# ## 3、如何构建Window MHSA并计算其复杂度？
# - 将patch组成的图片进一步划分成一个个更大的window
#     - 首先需要将三维的patch embedding转换成图片格式
#     - 使用unfold来将patch划分成window
# - 在每个window内部计算MHSA
#     - window数目其实可以跟batchsize进行统一对待，因为window与window之间没有交互计算
#     - 关于计算复杂度
#         - 假设窗的边长为W，那么计算每个窗的总体复杂度是$4W^2C^2+2W^4C$
#         - 假设patch的总数目为L，那么窗的数目为$L/W^2$
#         - 因此，W-MHSA的总体复杂度为$4LC^2+2LW^2C$
#     - 此处不需要mask
#     - 将计算结果转换成带window的四维张量格式
# - 复杂度对比
#     - **MHSA**: $4LC^2+2L^2C$
#     - **W-MHSA**：$4LC^2+2LW^2C$

# In[3]:


def window_multi_head_self_attention(patch_embedding, mhsa, window_size=4, num_head=num_head):
    num_patch_in_window = window_size * window_size
    bs, num_patch, patch_depth = patch_embedding.shape
    image_height = image_width = int(math.sqrt(num_patch))
    
    patch_embedding = patch_embedding.transpose(-1, -2)
    patch = patch_embedding.reshape(bs, patch_depth, image_height, image_width)
    window = F.unfold(patch, kernel_size=(window_size, window_size),
                      
                      stride=(window_size, window_size)).transpose(-1, -2) #[bs, num_window, window_depth]
    
    bs, num_window, patch_depth_times_num_patch_in_window = window.shape
    window = window.reshape(bs*num_window, patch_depth, num_patch_in_window).transpose(-1, -2) #[bs*num_w, num_patch, patch_depth]
    
    attn_prob, output = mhsa(window) #[bs*num_window, num_patch_in_window, patch_depth]
    
    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    return output


# ## 4、如何构建Shift Window MHSA及其Mask？
# - 将上一步的W-MHSA的结果转换成图片格式
# - 假设已经做了新的window划分，这一步叫做shift-window
# - 为了保持window数目不变从而有高效的计算，需要将图片的patch往左和往上各自滑动半个窗口大小的步长，保持patch所属window类别不变
# - 将图片patch还原成window的数据格式
# - 由于cycle shift后，每个window虽然形状规整，但部分window中存在原本不属于同一个窗口的patch，所以需要生成mask
# - 如何生成mask？
#     - 首先构建一个shift-window的patch所属的window类别矩阵
#     - 对该矩阵进行同样的往左和往上各自滑动半个窗口大小的步长的操作
#     - 通过unfold操作得到[bs, num_window, num_patch_in_window]形状的类别矩阵
#     - 对该矩阵进行扩维成[bs, num_window, num_patch_in_window, 1]
#     - 将该矩阵与其转置矩阵进行作差，得到同类关系矩阵（为0的位置上的patch属于同类，否则属于不同类）
#     - 对同类关系矩阵中非零的位置用负无穷数进行填充，对于零的位置用0去填充，这样就构建好了MHSA所需要的mask
#     - 此mask的形状为[bs, num_window, num_patch_in_window, num_patch_in_window]
# - 将window转换成三维的格式，[bs\*num_window, num_patch_in_window, patch_depth]
# - 将三维格式的特征连同mask一起送入MHSA中计算得到注意力输出
# - 将注意力输出转换成图片patch格式，[bs, num_window, num_patch_in_window, patch_depth]
# - 为了恢复位置，需要将图片的patch往右和往下各自滑动半个窗口大小的步长，至此，SW-MHSA计算完毕
# 

# In[4]:


# 定义一个辅助函数, window2image，也就是将transformer block的结果转化成图片的格式
def window2image(msa_output):
    bs, num_window, num_patch_in_window, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_height = int(math.sqrt(num_window)) * window_size
    image_width = image_height
    
    msa_output = msa_output.reshape(bs, int(math.sqrt(num_window)),
                                        int(math.sqrt(num_window)),
                                        window_size,
                                        window_size,
                                        patch_depth)
    msa_output = msa_output.transpose(2, 3)
    image = msa_output.reshape(bs, image_height*image_width, patch_depth)
    
    image = image.transpose(-1, -2).reshape(bs, patch_depth, image_height, image_width) # 跟卷积格式一致
    
    return image

# 定义辅助函数 shift_window，即高效地计算swmsa
def shift_window(w_msa_output, window_size, shift_size, generate_mask=False):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
    
    w_msa_output = window2image(w_msa_output) #[bs, depth, h, w]
    bs, patch_depth, image_height, image_width = w_msa_output.shape
    
    rolled_w_msa_output = torch.roll(w_msa_output, shifts=(shift_size, shift_size), dims=(2, 3))
    
    shifted_w_msa_input = rolled_w_msa_output.reshape(bs, patch_depth, 
                                                      int(math.sqrt(num_window)),
                                                      window_size,
                                                      int(math.sqrt(num_window)), 
                                                      window_size
                                                     )
    
    shifted_w_msa_input = shifted_w_msa_input.transpose(3, 4)
    shifted_w_msa_input = shifted_w_msa_input.reshape(bs, patch_depth, num_window*num_patch_in_window)
    shifted_w_msa_input = shifted_w_msa_input.transpose(-1, -2) # [bs, num_window*num_patch_in_window, patch_depth]
    shifted_window = shifted_w_msa_input.reshape(bs, num_window, num_patch_in_window, patch_depth)
    
    if generate_mask:
        additive_mask = build_mask_for_shifted_wmsa(bs, image_height, image_width, window_size)
    else:
        additive_mask = None
    
    return shifted_window, additive_mask


# In[16]:


# 构建shift window multi-head attention mask
def build_mask_for_shifted_wmsa(batch_size, image_height, image_width, window_size):
    index_matrix = torch.zeros(image_height, image_width)

    for i in range(image_height):
        for j in range(image_width):
            row_times = (i+window_size//2) // window_size
            col_times = (j+window_size//2) // window_size
            index_matrix[i, j] = row_times*(image_height//window_size) + col_times + 1
    rolled_index_matrix = torch.roll(index_matrix, shifts=(-window_size//2, -window_size//2), dims=(0, 1))
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0) #[bs, ch, h, w]
    
    c = F.unfold(rolled_index_matrix, kernel_size=(window_size, window_size),
                 stride=(window_size, window_size)).transpose(-1, -2) 
    
    c = c.tile(batch_size, 1, 1) #[bs, num_window, num_patch_in_window]
    
    bs, num_window, num_patch_in_window = c.shape
   
    c1 = c.unsqueeze(-1) # [bs, num_window, num_patch_in_window, 1]
    c2 = (c1 - c1.transpose(-1, -2)) == 0 #[bs, num_window, num_patch_in_window, num_patch_in_window]
    valid_matrix = c2.to(torch.float32)    
    additive_mask = (1-valid_matrix)*(-1e-9) #[bs, num_window, num_patch_in_window, num_patch_in_window]
    
    additive_mask = additive_mask.reshape(bs*num_window, num_patch_in_window, num_patch_in_window)
    
    return additive_mask
    


# In[5]:



def shift_window_multi_head_self_attention(w_msa_output, mhsa, window_size=4, num_head=num_head):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
    
    shifted_w_msa_input, additive_mask = shift_window(w_msa_output, window_size,
                                                      shift_size=-window_size//2,
                                                      generate_mask=True)

    #print(shifted_w_msa_input.shape) # [bs, num_window, num_patch_in_window, patch_depth]
    #print(additive_mask.shape) # [bs*num_window, num_patch_in_window, num_patch_in_window]

    shifted_w_msa_input = shifted_w_msa_input.reshape(bs*num_window, num_patch_in_window, patch_depth)
   
    attn_prob, output = mhsa(shifted_w_msa_input, additive_mask=additive_mask)
    
    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    
    output, _ = shift_window(output, window_size, shift_size=window_size//2, generate_mask=False)
    #print(output.shape) #[bs, num_window, num_patch_in_window, patch_depth]
    return output


# ## 5、如何构建Patch Merging？
# - 将window格式的特征转换成图片patch格式
# - 利用unfold操作，按照merge_size\*merge_size的大小得到新的patch, 形状为[bs, num_patch_new, merge_size\*merge_size\*patch_depth_old]
# - 使用一个全连接层对depth进行降维成0.5倍，也就是从merge_size\*merge_size\*patch_depth_old 映射到 0.5\*merge_size\*merge_size\*patch_depth_old
# - 输出的是patch embedding的形状格式, [bs, num_patch, patch_depth]
# - 举例说明：以merge_size=2为例，经过PatchMerging后，patch数目减少为之前的1/4，但是depth增大为原来的2倍，而不是4倍

# In[6]:


# 难点4 patch merging 
class PatchMerging(nn.Module):
    
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5):
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(
            model_dim*merge_size*merge_size,
            int(model_dim*merge_size*merge_size*output_depth_scale)
        )
        
    def forward(self, input):
        bs, num_window, num_patch_in_window, patch_depth = input.shape
        window_size = int(math.sqrt(num_patch_in_window))

        input = window2image(input) #[bs, patch_depth, image_h, image_w]

        merged_window = F.unfold(input, kernel_size=(self.merge_size, self.merge_size),
                                 stride=(self.merge_size, self.merge_size)
                                ).transpose(-1, -2)
        merged_window = self.proj_layer(merged_window) #[bs, num_patch, new_patch_depth]

        return merged_window
    


# ## 6、如何构建SwinTransformerBlock？
# - 每个block包含LayerNorm、W-MHSA、MLP、SW-MHSA、残差连接等模块
# - 输入是patch embedding格式
# - 每个MLP包含两层，分别是4\*model_dim和model_dim的大小
# - 输出的是window的数据格式，[bs, num_window, num_patch_in_window, patch_depth]
# - 需要注意残差连接对数据形状的要求




class SwinTransformerBlock(nn.Module):
    
    def __init__(self, model_dim, window_size, num_head, act_layer=nn.GELU, drop=0.):
        super(SwinTransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(model_dim) #model_dim=8
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.layer_norm4 = nn.LayerNorm(model_dim)

        
        self.wsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.wsma_mlp2 = nn.Linear(4*model_dim, model_dim)
        self.act = act_layer()
        self.swsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.swsma_mlp2 = nn.Linear(4*model_dim, model_dim)
        self.drop = nn.Dropout(drop)

        self.mhsa1 = MultiHeadSelfAttention(model_dim, num_head)
        self.mhsa2 = MultiHeadSelfAttention(model_dim, num_head)

        
    def forward(self, input):
        
        bs, num_patch, patch_depth = input.shape
        input = input.float()

        input1 = self.layer_norm1(input)
        input = input.long()
        w_msa_output = window_multi_head_self_attention(input1, self.mhsa1, window_size=4, num_head=num_head) #(8,4,16,64)
        bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
        w_msa_output = input + w_msa_output.reshape(bs, num_patch, patch_depth) #输入三维 + reshape后的三维
        output1 = self.act(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        output1 = self.act(self.wsma_mlp2(output1))
        output1 += w_msa_output

        input2 = self.layer_norm3(output1)
        input2 = input2.reshape(bs, num_window, num_patch_in_window, patch_depth)
        sw_msa_output = shift_window_multi_head_self_attention(input2, self.mhsa2, window_size=4, num_head=num_head)
        sw_msa_output = output1 + sw_msa_output.reshape(bs, num_patch, patch_depth)
        output2 = self.act(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        output2 = self.act(self.swsma_mlp2(output2))
        output2 += sw_msa_output

        output2 = output2.reshape(bs, num_window, num_patch_in_window, patch_depth)

        return output2


# ## 7、如何构建SwinTransformerModel？
# - 输入是图片
# - 首先对图片进行分块并得到Patch embedding
# - 经过第一个stage
# - 进行patch merging，再进行第二个stage
# - 以此类推...
# - 对最后一个block的输出转换成patch embedding的格式,[bs, num_patch, patch_depth]
# - 对patch embedding在时间维度进行平均池化，并映射到分类层得到分类的logits，完毕

# In[8]:


class SwinTransformerModel(nn.Module):
    
    def __init__(self, patch_size=4, model_dim_C=64, num_classes=10,
                 window_size=4, num_head=num_head, merge_size=2):
        
        super(SwinTransformerModel, self).__init__()
        patch_depth = patch_size*patch_size*3
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        #self.src_emb = nn.Embedding(src_len, d_model)
        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth, model_dim_C)) 
        
        self.block1 = SwinTransformerBlock(model_dim_C, window_size, num_head)
        self.block2 = SwinTransformerBlock(model_dim_C*2, window_size, num_head)
        self.block3 = SwinTransformerBlock(model_dim_C*4, window_size, num_head)
        self.block4 = SwinTransformerBlock(model_dim_C*8, window_size, num_head)

        
        self.patch_merging1 = PatchMerging(model_dim_C, merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C*2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C*4, merge_size)

        # self.final_layer = nn.Linear(model_dim_C*4, num_classes)
        self.final_layer = nn.Linear(model_dim_C * 64, num_classes)

        
    def forward(self, image):
        # patch_embedding_naive = image2emb_naive(image, self.patch_size, self.patch_embedding_weight)
        #image = self.src_emb(image)  # [batch_size, src_len, d_model]
        # patch_embedding_naive = image.reshape(16, 1024, 8)
        #print(patch_embedding_naive)
        
        #kernel = self.patch_embedding_weight.transpose(0, 1).reshape((-1, ic, patch_size, patch_size)) # oc*ic*kh*kw
        #patch_embedding_conv = image2emb_conv(image, kernel, self.patch_size) # 二维卷积的方法得到embedding
        #print(patch_embedding_conv)
        
        # block1
        patch_embedding = image
        # print(patch_embedding.shape)
        
        sw_msa_output = self.block1(patch_embedding)
        # print("block1_output", sw_msa_output.shape) #[bs, num_window, num_patch_in_window, patch_depth]

        merged_patch1 = self.patch_merging1(sw_msa_output)
        sw_msa_output_1 = self.block2(merged_patch1)
        # print("block2_output", sw_msa_output_1.shape)
        
        merged_patch2 = self.patch_merging2(sw_msa_output_1)
        sw_msa_output_2 = self.block3(merged_patch2)
        # print("block3_output", sw_msa_output_2.shape)
        
        #merged_patch3 = self.patch_merging3(sw_msa_output_2)
        #sw_msa_output_3 = self.block4(merged_patch3)
        # print("block4_output", sw_msa_output_3.shape)
        
        bs, num_window, num_patch_in_window, patch_depth = sw_msa_output_2.shape
        sw_msa_output_3 = sw_msa_output_2.reshape(bs, -1, patch_depth) #[bs, num_patch, patch_depth]
        # print("sw_msa_output_2", sw_msa_output_2.shape)
        #sw_msa_output_3 = sw_msa_output_3.reshape(bs, -1)  #[bs, patch_depth]
        # pool_output = torch.mean(sw_msa_output_2, dim=1)
        #logits = self.final_layer(sw_msa_output_3) #[bs, num_classes]
        # print("logits", logits.shape)
        #logits = nn.Softmax(dim=1)(logits)

        return sw_msa_output_3













