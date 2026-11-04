from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap

class TGTrackActor(BaseActor):
    """ Actor for training the tgtrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        self.cls_token = cfg.MODEL.ENCODER.CLASS_TOKEN
        self.use_generative_decoder = cfg.MODEL.GENERATIVE_DECODER.USE_GENERATIVE_DECODER

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        out_dict = self.forward_pass(data)

        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)
        template_frame_ids = data['template_frame_ids']
        outputs_list = []
        for i in range(len(search_list)):
            search_list_i = [search_list[i]]
            if self.multi_modal_language:
                text = data['nlp_ids'].permute (1,0)
                text_src = self.net(text_data=text, mode='text')
            else:
                text_src = None

            task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
            task_index_batch = torch.tensor(task_index_batch).cuda() #torch.Size([bs])

            enc_opt = self.net(template_list=template_list,
                               search_list=search_list_i,
                               template_anno_list=template_anno_list,
                               template_frame_ids=template_frame_ids,
                               text_src=text_src,
                               task_index=task_index_batch,
                               mode='encoder') # forward the encoder
            if self.cls_token:
                outputs, task_class_output = self.net(feature=enc_opt, mode="decoder_task_decoder")
                task_class_output = task_class_output.view(-1, task_class_output.size(-1))
                outputs['task_class'] = task_class_output
                outputs['task_class_label'] = task_index_batch
            if self.use_generative_decoder:
                outputs, generative_loss = self.net(feature=enc_opt, template_list=template_list,mode="decoder_generative_decoder")
                outputs['generative_loss'] = generative_loss
            else:
                outputs = self.net(feature=enc_opt, template_list=template_list, mode="decoder")

            outputs_list.append(outputs)

        return outputs_list

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        total_loss = torch.tensor(0.,dtype=torch.float).cuda()
        total_status = {}
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.ENCODER.STRIDE)  # list of torch.Size([b, H, W])
        for i in range(len(pred_dict)):
            if self.cls_token:
                task_cls_loss = self.objective['task_cls'](pred_dict[i]['task_class'], pred_dict[i]['task_class_label'])
            else:
                task_cls_loss = torch.tensor(0.,dtype=torch.float).cuda()

            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1) # torch.Size([b, 1, H, W])

            pred_boxes = pred_dict[i]['pred_boxes'] # torch.Size([b, 1, 4])
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            if self.use_generative_decoder:
                generative_loss = pred_dict[i]['generative_loss']
            else:
                generative_loss = torch.tensor(0.,dtype=torch.float).cuda()
            loss = (self.loss_weight['giou'] * giou_loss +
                    self.loss_weight['l1'] * l1_loss +
                    self.loss_weight['focal'] * location_loss +
                    self.loss_weight['task_cls'] * task_cls_loss +
                    self.loss_weight['gen'] * generative_loss)
            total_loss += loss
            if return_status:
                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                          f"{i}frame_Loss/giou": giou_loss.item(),
                          f"{i}frame_Loss/l1": l1_loss.item(),
                          f"{i}frame_Loss/location": location_loss.item(),
                          f"{i}frame_Loss/task_class": task_cls_loss.item(),
                          f"{i}frame_Loss/generative_class": generative_loss.item(),
                          f"{i}frame_IoU": mean_iou.item()}
                total_status.update(status)
        if return_status:
            return total_loss, total_status
        else:
            return total_loss