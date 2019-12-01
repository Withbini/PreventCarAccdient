#define _CRT_SECURE_NO_WARNINGS
#include "mylist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// ====================================================================
// ====================================================================

//JB addNodeed
float Getdistance(float x1, float x2, float y1, float y2) {
	float re;

	re = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
	printf("distance : %f ", re);
	return re;
}
LinkedList * newList() {
	LinkedList *lk = (LinkedList*)malloc(sizeof(LinkedList));
	lk->head = NULL;
	lk->tail = NULL;
	lk->size = 0;
	return lk;
}

void delete_List(LinkedList *lk) {
	Node *cur = lk->head;
	while (cur != NULL) {
		Node *nxt = cur->next;
		free(cur);
		cur = nxt;
	}
	free(lk);
	//puts("-------------------------------------------------------");
	//printf("%40s\n", "Linked List Destructed.");
	//puts("-------------------------------------------------------");
}

void delete_Nodes(LinkedList *lk) {
	Node *cur = lk->head;
	while (cur != NULL) {
		Node *nxt = cur->next;
		free(cur);
		cur = nxt;
	}
	lk->head = NULL;
	lk->tail = NULL;
	lk->size = 0;
}

int size(LinkedList *lk) {
	return lk->size;
}

void addNode(LinkedList *lk, frameData *data) {
	Node* tmp = (Node*)malloc(sizeof(Node));
	tmp->data = data;
	tmp->next = NULL;

	if (lk->head == NULL)
		lk->head = tmp;
	else
		lk->tail->next = tmp;
	lk->tail = tmp;
	++lk->size;
}

void insert_Node(LinkedList *lk, int n, frameData * data) {
	if (n == size(lk) + 1) // �� ���� �����Ѵٸ� �׳� addNode()�� ȣ���Ѵ�
		addNode(lk, data);
	else if (n < 1 || n > size(lk) + 1) // ����Ʈ ���� ���̶��
		printf("�ش� ��ġ(%d)�� ��带 ������ �� �����ϴ�.\n", n);
	else { // �� ���� �ƴ� ����Ʈ ���� ���� �ٸ� ���̶��
		Node* tmp = (Node*)malloc(sizeof(Node));
		tmp->data = data;

		if (n == 1) { // �׷��� ���� �� ���̶��
			tmp->next = lk->head; // head �����Ͱ� ����Ű�� �� ó�� ��带 tmp ������ ����
			lk->head = tmp; // head �����Ͱ� tmp�� ����Ű���� ����
		}
		else { // �� �յ� �ƴ϶��
			Node *cur = lk->head;
			while (--n - 1) // �ݺ������� ���� ��ġ ���� ��忡 ����
				cur = cur->next;

			tmp->next = cur->next; // tmp�� ������ cur�� ���� ��带 ����
			cur->next = tmp; // cur�� ���� ��带 tmp�� ����
		}
		++lk->size;
	}
}

void deleteNode(LinkedList *lk, int n) {
	if (n < 1 || n > size(lk)) // ����Ʈ ���� ���̶��
		printf("�ش� ��ġ(%d)�� ��带 ������ �� �����ϴ�.\n", n);
	else { // ����Ʈ ���� ���̶��
		Node *tmp;
		if (n == 1) { // �� �� ��带 �����ҰŶ��
			tmp = lk->head; // �� �� ��� �ּҸ� tmp�� ����
			lk->head = lk->head->next; // head �����Ͱ� �� �� ��� ������ ����Ű���� ����
			if (n == size(lk)) lk->tail = NULL; // ������ ���� ���� tail �����͸� NULL�� ����
		}
		else { // �� ���� �ƴ϶��
			Node *cur = lk->head; // cur�� �� ó�� ���� �������
			int i = n;
			while (--i - 1) // ������ ��� �������� ã�ư���
				cur = cur->next;

			tmp = cur->next; // ������ ���� cur ������ ����̹Ƿ� �̸� tmp�� ����
			cur->next = cur->next->next; // cur�� next �����Ϳ� cur�� ���� ���� ��带 ����
			if (n == size(lk)) lk->tail = cur; // ���� ��� �����̸� tail �����͸� �����Ѵ�
		}
		free(tmp->data);
		free(tmp); // �����Ϸ��� ���� �޸𸮸� ������Ų��
		--lk->size;
	}
}

void delNode(LinkedList *lk) {
	Node *cur = lk->head;
	int count = 0;
	while (cur != NULL && count <= lk->size) { // ��� ������ Ž���Ѵ�
		count++;
		if ((cur->data->check) == 0) { // ��ġ�ϴ� �����͸� ã�Ҵٸ�
			Node *next = cur;
			deleteNode(lk, count);
			cur = next;
			next = NULL;
		}
		else cur = cur->next;
	}
	// ��ġ�ϴ� �����Ͱ� ���ٸ�
	return;
}
void initNode(LinkedList * lk) {
	Node *cur = lk->head;
	while (cur != NULL) {
		cur->data->check = 0;
		cur = cur->next;
	}
}

//*********************************************************struct�� �ٲܿ���
////����Ʈ�� ������� ����, �ڵ��� ���� �ִ� ����Ʈ�� Ž��
//int search(LinkedList *ref_lk, LinkedList *cur_lk, char *LabelName, float x1, float y1, int threshold, int objcount) {
//	Node *cur = ref_lk->head;
//	int n = ref_lk->size;
//
//	int NewCarcnt = 0;
//	while (cur != NULL && n--) { // ��� ������ Ž���Ѵ�
//		if (!strcmp(cur->data->LabelName, LabelName) && cur->data->check == 0) {
//			float dist;
//			if (dist=Getdistance(x1, cur->data->x, y1, cur->data->y)< threshold) { //�Ÿ��� thresh ���ϸ� ���� ��ü->���� ī��Ʈ �ű��
//				cur->data->check = 1;
//				frameData *Newdata = (frameData*)malloc(sizeof(frameData));
//				strcpy(Newdata->LabelName, LabelName);
//				Newdata->LabelNum = cur->data->LabelNum;
//				Newdata->x = x1;
//				Newdata->y = y1;
//				Newdata->check = 0;
//				addNode(cur_lk, Newdata);
//				printf("everything is same~~~~~~~~~~~~~~~\n");
//				break;
//			}
//			else {
//				printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~distance : %f", dist);
//			}
//		}
//		cur = cur->next;
//	}
//	// �ڵ��� ���̸鼭 Ž�� ���ߴµ��� ������ ���ο� ��ü��� �Ǵ�
//	// ���ο� �Ϳ� �ֱ�
//	if (cur == NULL && !strcmp(LabelName, "car")) {
//		NewCarcnt++;
//		frameData *Newdata = (frameData*)malloc(sizeof(frameData));
//		strcpy(Newdata->LabelName, LabelName);
//		Newdata->LabelNum = NewCarcnt;
//		Newdata->x = x1;
//		Newdata->y = y1;
//		Newdata->check = 0;
//		printf("new Label!!!!!!!!!!!!!!!!!!!!!!!!!! %d\n", Newdata->LabelNum);
//		addNode(cur_lk, Newdata);
//	}
//	objcount += NewCarcnt;
//	return objcount;
//
//}
int search(LinkedList *ref_lk, LinkedList *cur_lk, char *LabelName, float x1, float y1, int threshold, int objcount, int *labelnum) {
	Node *cur = ref_lk->head;
	int n = ref_lk->size;
	printf("n is %d\n", n);
	int NewCarcnt = 0;
	while (cur != NULL) { // ��� ������ Ž���Ѵ�
		if (!strcmp(cur->data->LabelName, LabelName) && cur->data->check == 0) {
			printf("exist~~~\n");
			if (Getdistance(x1, cur->data->x, y1, cur->data->y)< threshold) { //�Ÿ��� thresh ���ϸ� ���� ��ü->���� ī��Ʈ �ű��
				printf("same~~~~\n");
				cur->data->check = 1;
				frameData *Newdata = (frameData*)malloc(sizeof(frameData));
				strcpy(Newdata->LabelName, LabelName);
				Newdata->LabelNum = cur->data->LabelNum;
				Newdata->x = x1;
				Newdata->y = y1;
				Newdata->check = 0;
				addNode(cur_lk, Newdata);
				printf("everything is same~~~~~~~~~~~~~~~\n");
				break;
			}
		}
		cur = cur->next;
	}
	printf("new~~~~~~\n");
	// �ڵ��� ���̸鼭 Ž�� ���ߴµ��� ������ ���ο� ��ü��� �Ǵ�
	// ���ο� �Ϳ� �ֱ�
	if (cur == NULL && !strcmp(LabelName, "car")) {
		NewCarcnt++;
		frameData *Newdata = (frameData*)malloc(sizeof(frameData));
		strcpy(Newdata->LabelName, LabelName);
		Newdata->LabelNum = NewCarcnt;
		Newdata->x = x1;
		Newdata->y = y1;
		Newdata->check = 0;
		printf("new Label!!!!!!!!!!!!!!!!!!!!!!!!!! %d\n", Newdata->LabelNum);
		addNode(cur_lk, Newdata);
	}
	initNode(cur_lk);
	objcount += NewCarcnt;
	//printf("before copy %d \t", *labelnum);
	memcpy(labelnum, &NewCarcnt, sizeof(int));
	//printf("after copy %d\n", *labelnum);
	printf("cur lk size before %d\n", cur_lk->size);
	return objcount;

}
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

// ====================================================================
// ====================================================================
