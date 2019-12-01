#ifndef _MY_LIST_H
#define _MY_LIST_H
// ====================================================================
// ====================================================================
#ifdef __cplusplus
extern "C" {
#endif

	typedef struct DATA {
		char LabelName[128];
		int LabelNum;
		float x, y;
		int check;
	}frameData;

	typedef struct Node //��� ����
	{
		frameData* data;
		struct Node *next;
	}Node;

	typedef struct LinkedList {
		Node *head; // �Ӹ� ���
		Node *tail; // ���� ���
		int size; // ���Ḯ��Ʈ ũ��
	} LinkedList;

	typedef struct list_key {
		char *LabelName;
		float x;
		float y;
		int threshold;
	}listKey;

	typedef struct list_res {
		int objcount;
		int LabelNum;
	}listRes;


	//JB addNodeed
	int Getdistance(float x1, float x2, float y1, float y2, int thresh);
	LinkedList * newList();

	void delete_List(LinkedList *lk);

	int size(LinkedList *lk);

	void addNode(LinkedList *lk, frameData *data);

	void insert_Node(LinkedList *lk, int n, frameData * data);

	void deleteNode(LinkedList *lk, int n);

	void delNode(LinkedList *lk);
	void delete_Nodes(LinkedList *lk);

	void initNode(LinkedList * lk);

	int search(LinkedList *ref_lk, LinkedList *cur_lk, char *LabelName, float x1, float y1, int threshold, int objcount, int *labelnum);

	//int main() {
	//	LinkedList *list = newList();
	//
	//	char label[4000] = "���¹ٺ�";
	//	int thresh = 50;
	//	search(list, "asdf", 10, 10, thresh, label); //�߰�
	//
	//												 //���� ����
	//	search(list, "asdf", 200, 200, thresh, label); //�߰�
	//
	//	search(list, "asdf", 300, 300, thresh, label); //�߰�
	//	delNode(list);
	//
	//
	//	//���� ������
	//	Node *cur = list->head;
	//	while (cur != NULL) {
	//		cur->data->check = 0;
	//		cur = cur->next;
	//	}
	//	search(list, "asdf", 10, 10, thresh, label); //Ž����� ����->������ ��������
	//
	//	search(list, "asdf", 100, 100, thresh, label); //�߰�
	//
	//	delNode(list);//30����
	//
	//	return 0;
	//}

#ifdef __cplusplus
}
#endif
// ====================================================================
// ====================================================================


#endif