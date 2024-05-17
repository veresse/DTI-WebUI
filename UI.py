# 导入所需库和模块
from gradio import Interface, components
from model2 import test2, preprocess_data2  # 导入模型2相关函数
from model1 import test1, preprocess_data1  # 导入模型1相关函数
import json
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, MolToImage
import io
from PIL import Image
import requests
import time
import gzip
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# 定义一个类，用于解析PDB文件
class PDBParser:
    def __init__(self, pdb_content):
        self.pdb_content = pdb_content
        self.atoms = []

    # 解析PDB内容
    def parse(self):
        lines = self.pdb_content.decode('utf-8').split('\n')
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_data = self._parse_atom_line(line)
                self.atoms.append(atom_data)

    # 解析PDB文件中的原子行
    def _parse_atom_line(self, line):
        atom_name = line[12:16].strip()
        atom_serial = int(line[6:11].strip())
        atom_element = line[76:78].strip()
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        return {
            'atom_name': atom_name,
            'atom_serial': atom_serial,
            'atom_element': atom_element,
            'x': x,
            'y': y,
            'z': z
        }

# 定义一个类，用于可视化PDB文件中的原子信息
class PDBVisualizer:
    def __init__(self, atoms):
        self.atoms = atoms

    # 可视化PDB文件中的原子信息
    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for atom in self.atoms:
            ax.scatter(atom['x'], atom['y'], atom['z'], label=atom['atom_name'])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        png_image = Image.open(img_buffer)

        return png_image

# 定义一个类，用于预测蛋白质的结构
class ProteinStructurePredictor:
    def __init__(self):
        # 将 API 令牌硬编码到代码中
        self.api_token = "b24ab28e5a73c9b93e29fd4f26daa9d535812509"

    # 预测蛋白质的结构
    def predict_structure(self, target_sequence):
        project_title = "Protein Model"

        response = requests.post(
            "https://swissmodel.expasy.org/automodel",
            headers={"Authorization": f"Token {self.api_token}"},
            json={
                "target_sequences": [target_sequence],
                "project_title": project_title
            }
        )

        # 若请求成功，返回项目 ID；否则打印失败信息并退出程序
        if response.status_code == 202:
            print("建模请求已接受，项目正在排队等待执行...")
            project_id = response.json()["project_id"]
        else:
            print(f"请求失败，状态码：{response.status_code}")
            print("响应内容：", response.text)
            exit()

        # 循环查询项目状态，直到完成或失败为止
        while True:
            time.sleep(10)
            status_response = requests.get(
                f"https://swissmodel.expasy.org/project/{project_id}/models/summary/",
                headers={"Authorization": f"Token {self.api_token}"}
            )

            status = status_response.json()["status"]

            print(f"当前项目状态：{status}")

            if status in ["COMPLETED", "FAILED"]:
                break

        # 若项目完成，解析模型坐标并返回可视化的图片
        if status == "COMPLETED":
            response_json = status_response.json()
            for model in response_json['models']:
                pdb_url = model['coordinates_url']
                print(f"模型坐标URL：{pdb_url}")

                pdb_content_response = requests.get(pdb_url)
                pdb_content = gzip.decompress(pdb_content_response.content)

                pdb_parser = PDBParser(pdb_content)
                pdb_parser.parse()
                atoms_data = pdb_parser.atoms

                visualizer = PDBVisualizer(atoms_data)
                png_image = visualizer.visualize()

                return png_image

# 加载配置文件
def load_config():
    with open(".\config\config.json", "r") as f:
        config = json.load(f)
    return config["model_paths"]

model_paths = load_config()

# 根据SMILES字符串绘制分子结构图像
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)

    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png_bytes = drawer.GetDrawingText()

    png_image = Image.open(io.BytesIO(png_bytes))

    return png_image

# 使用模型1进行预测
def model1(drug_smiles, protein_sequence,model_path=model_paths["model1"]):
    atom_feature, adj, smi_id, prot_id = preprocess_data1(drug_smiles, protein_sequence)
    predicted_labels = test1(atom_feature, adj, smi_id, prot_id, model_path)
    interaction_predictions = "无法发生交互作用" if predicted_labels[0, 0] > 0 else "可以发生交互作用" if predicted_labels[0, 1] > 0 else "无法发生交互作用"
    return interaction_predictions

# 使用模型2进行预测
def model2(drug_smiles, protein_sequence,model_path=model_paths["model2"]):
    atom_feature, adj, smi_id, prot_id = preprocess_data2(drug_smiles, protein_sequence)
    predicted_labels = test2(atom_feature, adj, smi_id, prot_id, model_path)
    interaction_predictions = "无法发生交互作用" if predicted_labels[0, 0] > 0 else "可以发生交互作用" if predicted_labels[0, 1] > 0 else "无法发生交互作用"
    return interaction_predictions

# 验证输入的药物和蛋白质序列是否有效
def validate_input(drug, protein):
    drug_smiles, protein_sequence = drug, protein
    if not drug_smiles or not protein_sequence:
        return "输入的药物序列或蛋白序列不能为空"

    return drug, protein

# 预测交互作用并生成相应的图像
def predict(model, drug, protein):
    validation_result = validate_input(drug, protein)

    if isinstance(validation_result, str) and validation_result.startswith("无效"):
        return validation_result

    drug, protein = validation_result

    interaction_predictions_text = None
    protein_structure_image_pil = None

    if model == "DPAG":
        interaction_predictions_text = model1(drug, protein, model_paths["model1"])
    elif model == "MNDT":
        interaction_predictions_text = model2(drug, protein, model_paths["model2"])

    protein_predictor = ProteinStructurePredictor()
    protein_structure_image_pil = protein_predictor.predict_structure(protein)
    molecular_formula = draw_molecule(drug)

    output = (interaction_predictions_text, protein_structure_image_pil, molecular_formula)

    return output if interaction_predictions_text is not None else "无法获取预测结果，请检查输入或模型选择"

# 定义输入组件
input_components = [
    components.Dropdown(choices=["DPAG", "MNDT"], label="选择模型"),
    components.Textbox(label="药物序列"),
    components.Textbox(label="蛋白序列"),
]

# 定义输出组件
output_components = [
    components.Textbox(label="预测结果"),
    components.Image(label="蛋白结构图", type='pil'),
    components.Image(label="分子结构图", type='pil'),
]

# 创建界面
interface = Interface(
    fn=lambda model, drug, protein: predict(model, drug, protein),
    inputs=input_components,
    outputs=output_components,
    title="药物-标靶交互作用预测系统"
)

# 启动界面
interface.launch(share=True)
