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
	if (n == size(lk) + 1) // 맨 끝에 삽입한다면 그냥 addNode()를 호출한다
		addNode(lk, data);
	else if (n < 1 || n > size(lk) + 1) // 리스트 범위 밖이라면
		printf("해당 위치(%d)에 노드를 삽입할 수 없습니다.\n", n);
	else { // 맨 끝이 아닌 리스트 범위 내의 다른 곳이라면
		Node* tmp = (Node*)malloc(sizeof(Node));
		tmp->data = data;

		if (n == 1) { // 그런데 하필 맨 앞이라면
			tmp->next = lk->head; // head 포인터가 가리키는 맨 처음 노드를 tmp 다음에 연결
			lk->head = tmp; // head 포인터가 tmp를 가리키도록 갱신
		}
		else { // 맨 앞도 아니라면
			Node *cur = lk->head;
			while (--n - 1) // 반복문으로 지정 위치 이전 노드에 접근
				cur = cur->next;

			tmp->next = cur->next; // tmp의 다음에 cur의 다음 노드를 연결
			cur->next = tmp; // cur의 다음 노드를 tmp로 갱신
		}
		++lk->size;
	}
}

void deleteNode(LinkedList *lk, int n) {
	if (n < 1 || n > size(lk)) // 리스트 범위 밖이라면
		printf("해당 위치(%d)의 노드를 삭제할 수 없습니다.\n", n);
	else { // 리스트 범위 안이라면
		Node *tmp;
		if (n == 1) { // 맨 앞 노드를 삭제할거라면
			tmp = lk->head; // 맨 앞 노드 주소를 tmp에 저장
			lk->head = lk->head->next; // head 포인터가 맨 앞 노드 다음을 가리키도록 갱신
			if (n == size(lk)) lk->tail = NULL; // 마지막 남은 노드면 tail 포인터를 NULL로 갱신
		}
		else { // 맨 앞이 아니라면
			Node *cur = lk->head; // cur을 맨 처음 노드로 맞춰놓고
			int i = n;
			while (--i - 1) // 삭제할 노드 직전까지 찾아간다
				cur = cur->next;

			tmp = cur->next; // 삭제할 노드는 cur 다음의 노드이므로 이를 tmp에 저장
			cur->next = cur->next->next; // cur의 next 포인터에 cur의 다음 다음 노드를 연결
			if (n == size(lk)) lk->tail = cur; // 꼬리 노드 삭제이면 tail 포인터를 갱신한다
		}
		free(tmp->data);
		free(tmp); // 삭제하려는 노드는 메모리를 해제시킨다
		--lk->size;
	}
}

void delNode(LinkedList *lk) {
	Node *cur = lk->head;
	int count = 0;
	while (cur != NULL && count <= lk->size) { // 노드 끝까지 탐색한다
		count++;
		if ((cur->data->check) == 0) { // 일치하는 데이터를 찾았다면
			Node *next = cur;
			deleteNode(lk, count);
			cur = next;
			next = NULL;
		}
		else cur = cur->next;
	}
	// 일치하는 데이터가 없다면
	return;
}
void initNode(LinkedList * lk) {
	Node *cur = lk->head;
	while (cur != NULL) {
		cur->data->check = 0;
		cur = cur->next;
	}
}

//*********************************************************struct로 바꿀예정
////리스트를 순서대로 돌며, 자동차 라벨이 있는 리스트를 탐색
//int search(LinkedList *ref_lk, LinkedList *cur_lk, char *LabelName, float x1, float y1, int threshold, int objcount) {
//	Node *cur = ref_lk->head;
//	int n = ref_lk->size;
//
//	int NewCarcnt = 0;
//	while (cur != NULL && n--) { // 노드 끝까지 탐색한다
//		if (!strcmp(cur->data->LabelName, LabelName) && cur->data->check == 0) {
//			float dist;
//			if (dist=Getdistance(x1, cur->data->x, y1, cur->data->y)< threshold) { //거리가 thresh 이하면 같은 물체->새로 카운트 매기기
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
//	// 자동차 라벨이면서 탐색 다했는데도 없으면 새로운 물체라고 판단
//	// 새로운 것에 넣기
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
	while (cur != NULL) { // 노드 끝까지 탐색한다
		if (!strcmp(cur->data->LabelName, LabelName) && cur->data->check == 0) {
			printf("exist~~~\n");
			if (Getdistance(x1, cur->data->x, y1, cur->data->y)< threshold) { //거리가 thresh 이하면 같은 물체->새로 카운트 매기기
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
	// 자동차 라벨이면서 탐색 다했는데도 없으면 새로운 물체라고 판단
	// 새로운 것에 넣기
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
//	char label[4000] = "저는바보";
//	int thresh = 50;
//	search(list, "asdf", 10, 10, thresh, label); //추가
//
//												 //다음 사진
//	search(list, "asdf", 200, 200, thresh, label); //추가
//
//	search(list, "asdf", 300, 300, thresh, label); //추가
//	delNode(list);
//
//
//	//다음 프레임
//	Node *cur = list->head;
//	while (cur != NULL) {
//		cur->data->check = 0;
//		cur = cur->next;
//	}
//	search(list, "asdf", 10, 10, thresh, label); //탐색결과 있음->이전꺼 가져오기
//
//	search(list, "asdf", 100, 100, thresh, label); //추가
//
//	delNode(list);//30삭제
//
//	return 0;
//}

// ====================================================================
// ====================================================================
