import xml.etree.ElementTree as ET

# 解析XML文件
tree = ET.parse(r"D:\github_repository\Annotation_Human\Human Full Annotation_exp1_F0009 Data.xml")
root = tree.getroot()

def extract_info(node, parent_id='-1'):
    results = []
    # 遍历所有的<a>标签
    for a in node.findall('./a'):
        cell_id = a.get('id')
        child_ss = a.find('ss')

        # 如果找到了<ss>标签，则进一步提取<s>标签中的信息
        if child_ss is not None:
            for s in child_ss.findall('s'):
                frame_index = s.get('i')
                x_coord = s.get('x')
                y_coord = s.get('y')
                results.append((int(frame_index), cell_id, x_coord, y_coord, parent_id))

        # 递归处理嵌套的<a>标签
        child_as = a.find('as')
        if child_as is not None:
            results.extend(extract_info(child_as, cell_id))

    return results

# 提取name="Annotate DarkSpot Center"的<f>标签内的信息
def extract_from_f_tag(root):
    results = []

    # 遍历所有的<f>标签
    for f in root.findall('.//f[@name="Annotate DarkSpot Center"]'):
        as_tag = f.find('./as')
        if as_tag is not None:
            results.extend(extract_info(as_tag))

    return results

# 提取信息
info_list = extract_from_f_tag(root)

# 按帧序号排序
info_list_sorted = sorted(info_list, key=lambda x: x[0])

# 写入到TXT文件
with open(r"C:\Users\86137\Desktop\tracklet.txt", 'w') as f:
    for info in info_list_sorted:
        f.write(' '.join(map(str, info)) + '\n')