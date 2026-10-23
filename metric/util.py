from xml.etree.ElementTree import ElementTree


def get_ann_info_from_xml(cur_ann_path):
        tree = read_xml(cur_ann_path)
        filename_nodes = get_node_by_keyvalue(find_nodes(tree, "filename"), {})
        object_nodes = get_node_by_keyvalue(find_nodes(tree, "object"), {})
        name_nodes = get_node_by_keyvalue(find_nodes(tree, "object/name"), {})

        xmin_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/xmin"), {})
        ymin_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/ymin"), {})
        xmax_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/xmax"), {})
        ymax_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/ymax"), {})

        boxes = []
        for index, node in enumerate(object_nodes):
            xmin = int(xmin_nodes[index].text)
            ymin = int(ymin_nodes[index].text)
            xmax = int(xmax_nodes[index].text)
            ymax = int(ymax_nodes[index].text)
            
            label = name_nodes[index].text
            
            width = xmax - xmin
            height = ymax - ymin
            box = [xmin, ymin, xmax, ymax, label]
            boxes += [box]
        return boxes

def read_xml(in_path):
    '''读取并解析xml文件'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def if_match(node, kv_map):
    '''判断某个节点是否包含所有传入参数属性
      node: 节点
      kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


def get_node_by_keyvalue(nodelist, kv_map):
    '''根据属性及属性值定位符合的节点，返回节点
      nodelist: 节点列表
      kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


def find_nodes(tree, path):
    '''查找某个路径匹配的所有节点
      tree: xml树
      path: 节点路径'''
    return tree.findall(path)


def compute_iou(pred, gt):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], pred[0])
    yA = max(gt[1], pred[1])
    xB = min(gt[2], pred[2])
    yB = min(gt[3], pred[3])
    # if there is no overlap between predicted and ground-truth box
    if xB < xA or yB < yA:
        return 0.0
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou