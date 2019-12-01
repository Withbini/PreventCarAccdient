//#include "mylist.h"
//#include <stdio.h>
//struct asdf {
//	int *aa;
//};
//
//
//void asdf(struct asdf *a) {
//	(a->aa)++;
//	printf("%d ", a->aa);
//}
//int main() {
//	LinkedList *list = newList();
//	LinkedList *list2 = newList();
//	Node *cur = list2->head;
//	int i = 20;
//	int j = 0;
//	frameData *Newdata = (frameData*)malloc(sizeof(frameData));
//	strcpy(Newdata->LabelName, "JB");
//	Newdata->LabelNum = 1;
//	Newdata->x = 1;
//	Newdata->y = 1;
//	Newdata->check = 0;
//	addNode(list2, Newdata);
//	
//	//memcpy(list, list2, sizeof(LinkedList));
//	for (i = 0; i < 5; i++) {
//
//		Node *cur = list2->head;
//		while (cur != NULL) {
//			Node *node = (Node*)malloc(sizeof(Node));
//			memcpy(node->data, cur->data, sizeof(frameData));
//			if (cur->next != NULL) {
//				memcpy(node->next, cur->next, sizeof(Node));
//			}
//
//			cur = cur->next;
//		}
//
//		addNode(list2, Newdata);
//	}
//	system("pause");
//	return 0;
//}